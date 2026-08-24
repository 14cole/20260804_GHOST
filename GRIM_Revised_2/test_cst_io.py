"""CST wide-table and legacy .cst_data import regressions."""

import csv
import os
import tempfile
import unittest

import numpy as np

from grim_dataset import RcsGrid


class TestCstIo(unittest.TestCase):
    def _write(self, rows, suffix=".csv"):
        descriptor, path = tempfile.mkstemp(suffix=suffix)
        os.close(descriptor)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        with open(path, "w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)
        return path

    @staticmethod
    def _sample(grid, azimuth, elevation, frequency, polarization):
        ai = int(np.where(np.isclose(grid.azimuths, azimuth))[0][0])
        ei = int(np.where(np.isclose(grid.elevations, elevation))[0][0])
        fi = int(np.where(np.isclose(grid.frequencies, frequency))[0][0])
        pi = int(np.where(grid.polarizations == polarization)[0][0])
        return grid.rcs_power[ai, ei, fi, pi], grid.rcs_phase[ai, ei, fi, pi]

    def test_flat_cst_data_axes_units_polarity_and_iq_validation(self):
        path = self._write(
            [
                [
                    "Elevation(deg)", "Azimuth(deg)", "Frequency(GHz)",
                    "Polarity", "Magnitude(dBsm)", "Phase(deg)", "IQ",
                ],
                [10.0, 270.0, 9.0, "vv", 0.0, 90.0, "1i"],
                [-5.0, 180.0, 10.0, "HH", 6.020599913, -90.0, "-2i"],
            ],
            suffix=".cst_data",
        )

        grid = RcsGrid.read_CST(path)

        np.testing.assert_allclose(grid.azimuths, [-180.0, -90.0])
        np.testing.assert_allclose(grid.elevations, [-5.0, 10.0])
        np.testing.assert_allclose(grid.frequencies, [9.0, 10.0])
        np.testing.assert_array_equal(grid.polarizations, ["VV", "HH"])
        vv_power, vv_phase = self._sample(grid, -90.0, 10.0, 9.0, "VV")
        hh_power, hh_phase = self._sample(grid, -180.0, -5.0, 10.0, "HH")
        self.assertAlmostEqual(float(vv_power), 1.0, places=12)
        self.assertAlmostEqual(float(vv_phase), np.pi / 2.0, places=12)
        self.assertAlmostEqual(float(hh_power), 4.0, places=8)
        self.assertAlmostEqual(float(hh_phase), -np.pi / 2.0, places=12)
        self.assertEqual(grid.units["frequency"], "GHz")
        self.assertEqual(grid.extra["source_format"], "CST flat cst_data")
        self.assertEqual(grid.extra["cst_iq_rows_validated"], 2)

    def test_flat_cst_data_uses_explicit_magnitude_phase_when_iq_is_opaque(self):
        path = self._write(
            [
                [
                    "Elevations (deg)", "Azimuth (deg)", "Frequency (Hz)",
                    "Polarization", "Magnitude (dBsm)", "Phase (deg)", "IQ",
                ],
                [0.0, 0.0, 2.0e9, "VH", -10.0, 45.0, "vendor-token"],
            ],
            suffix=".cst_data",
        )

        grid = RcsGrid.read_CST(path)

        power, phase = self._sample(grid, 0.0, 0.0, 2.0, "VH")
        self.assertAlmostEqual(float(power), 0.1, places=12)
        self.assertAlmostEqual(float(phase), np.pi / 4.0, places=12)
        self.assertEqual(grid.extra["cst_iq_unparsed_fallback_rows"], 1)

    def test_flat_cst_data_rejects_iq_disagreement(self):
        magnitude_path = self._write(
            [
                [
                    "Elevation(deg)", "Azimuth(deg)", "Frequency(GHz)",
                    "Polarity", "Magnitude(dBsm)", "Phase(deg)", "IQ",
                ],
                [0.0, 0.0, 1.0, "VV", 0.0, 0.0, "2+0i"],
            ],
            suffix=".cst_data",
        )
        phase_path = self._write(
            [
                [
                    "Elevation(deg)", "Azimuth(deg)", "Frequency(GHz)",
                    "Polarity", "Magnitude(dBsm)", "Phase(deg)", "IQ",
                ],
                [0.0, 0.0, 1.0, "VV", 0.0, 90.0, "1+0i"],
            ],
            suffix=".cst_data",
        )

        with self.assertRaisesRegex(ValueError, "IQ magnitude disagrees"):
            RcsGrid.read_CST(magnitude_path)
        with self.assertRaisesRegex(ValueError, "IQ phase disagrees"):
            RcsGrid.read_CST(phase_path)

    def test_flat_cst_data_rejects_ambiguous_unitless_headers(self):
        path = self._write(
            [[
                "Elevation", "Azimuth", "Frequency", "Polarity",
                "Magnitude", "Phase", "IQ",
            ], [0.0, 0.0, 1.0, "VV", 0.0, 0.0, "1+0i"]],
            suffix=".cst_data",
        )

        with self.assertRaisesRegex(ValueError, "Could not find the .cst_data header"):
            RcsGrid.read_CST(path)

        thz_path = self._write(
            [[
                "Elevation(deg)", "Azimuth(deg)", "Frequency(THz)",
                "Polarity", "Magnitude(dBsm)", "Phase(deg)",
            ], [0.0, 0.0, 1.0, "VV", 0.0, 0.0]],
            suffix=".cst_data",
        )
        with self.assertRaisesRegex(ValueError, "Could not find the .cst_data header"):
            RcsGrid.read_CST(thz_path)

    def test_flat_cst_data_keeps_iq_precision_after_redundant_validation(self):
        path = self._write(
            [
                [
                    "Elevation(deg)", "Azimuth(deg)", "Frequency(GHz)",
                    "Polarity", "Magnitude(dBsm)", "Phase(deg)", "IQ",
                ],
                # The rounded display columns are within the accepted
                # agreement tolerance, but IQ remains the coherent authority.
                [0.0, 0.0, 1.0, "VV", 0.04, 0.4, "1+0i"],
            ],
            suffix=".cst_data",
        )

        grid = RcsGrid.read_CST(path)
        power, phase = self._sample(grid, 0.0, 0.0, 1.0, "VV")

        self.assertEqual(float(power), 1.0)
        self.assertEqual(float(phase), 0.0)

    def test_flat_cst_data_merges_matching_seam_and_rejects_conflict(self):
        matching_path = self._write(
            [
                [
                    "Elevation(deg)", "Azimuth(deg)", "Frequency(GHz)",
                    "Polarity", "Magnitude(dBsm)", "Phase(deg)",
                ],
                [0.0, -180.0, 1.0, "VV", 0.0, 0.0],
                [0.0, 180.0, 1.0, "VV", 0.0, 0.0],
            ],
            suffix=".cst_data",
        )

        matching = RcsGrid.read_CST(matching_path)
        np.testing.assert_allclose(matching.azimuths, [-180.0])
        self.assertEqual(matching.rcs_power.size, 1)

        conflicting_path = self._write(
            [
                [
                    "Elevation(deg)", "Azimuth(deg)", "Frequency(GHz)",
                    "Polarity", "Magnitude(dBsm)", "Phase(deg)",
                ],
                [0.0, -180.0, 1.0, "VV", 0.0, 0.0],
                [0.0, 180.0, 1.0, "VV", 3.0, 0.0],
            ],
            suffix=".cst_data",
        )
        with self.assertRaisesRegex(ValueError, "conflicting duplicate CST sample"):
            RcsGrid.read_CST(conflicting_path)

    def test_cst_dbsm_overflow_is_rejected_in_both_schemas(self):
        flat_path = self._write(
            [
                [
                    "Elevation(deg)", "Azimuth(deg)", "Frequency(GHz)",
                    "Polarity", "Magnitude(dBsm)", "Phase(deg)",
                ],
                [0.0, 0.0, 1.0, "VV", 4000.0, 0.0],
            ],
            suffix=".cst_data",
        )
        wide_path = self._write(
            [
                [
                    "Frequency(GHz)", "Theta(deg)", "Phi(deg)",
                    "RCS Theta-Theta(dBsm)", "Phase Theta-Theta(deg)",
                ],
                [1.0, 90.0, 0.0, 4000.0, 0.0],
            ]
        )
        iq_path = self._write(
            [
                [
                    "Elevation(deg)", "Azimuth(deg)", "Frequency(GHz)",
                    "Polarity", "IQ",
                ],
                [0.0, 0.0, 1.0, "VV", "1e308+0i"],
            ],
            suffix=".cst_data",
        )

        with self.assertRaisesRegex(ValueError, "overflows finite linear power"):
            RcsGrid.read_CST(flat_path)
        with self.assertRaisesRegex(ValueError, "overflows finite linear power"):
            RcsGrid.read_CST(wide_path)
        with self.assertRaisesRegex(ValueError, "overflows finite linear power"):
            RcsGrid.read_CST(iq_path)

    def test_cst_rejects_nonpositive_frequency_in_both_schemas(self):
        flat_path = self._write(
            [[
                "Elevation(deg)", "Azimuth(deg)", "Frequency(GHz)",
                "Polarity", "Magnitude(dBsm)", "Phase(deg)",
            ], [0.0, 0.0, 0.0, "VV", 0.0, 0.0]],
            suffix=".cst_data",
        )
        wide_path = self._write(
            [[
                "Frequency(GHz)", "Theta(deg)", "Phi(deg)",
                "RCS Theta-Theta(dBsm)", "Phase Theta-Theta(deg)",
            ], [-1.0, 90.0, 0.0, 0.0, 0.0]]
        )

        with self.assertRaisesRegex(ValueError, "frequency must be positive"):
            RcsGrid.read_CST(flat_path)
        with self.assertRaisesRegex(ValueError, "frequency must be positive"):
            RcsGrid.read_CST(wide_path)

    def test_wide_cst_converts_theta_and_discovers_only_present_pols(self):
        path = self._write(
            [
                ["CST Studio Suite farfield export"],
                [
                    "Frequency (Hz)", "Theta (deg)", "Phi (deg)",
                    "RCS Theta-Theta (dBsm)", "RCS Phi-Phi (dBsm)",
                    "Phase Theta-Theta (deg)", "Phase Phi-Phi (deg)",
                ],
                [1.0e9, 0.0, 270.0, 0.0, 6.020599913, 0.0, 90.0],
                [2.0e9, 180.0, 180.0, -10.0, -20.0, 180.0, -180.0],
            ]
        )

        grid = RcsGrid.read_CST(path)

        np.testing.assert_allclose(grid.azimuths, [-180.0, -90.0])
        np.testing.assert_allclose(grid.elevations, [-90.0, 90.0])
        np.testing.assert_allclose(grid.frequencies, [1.0, 2.0])
        np.testing.assert_array_equal(grid.polarizations, ["VV", "HH"])
        power, phase = self._sample(grid, -90.0, 90.0, 1.0, "HH")
        self.assertAlmostEqual(float(power), 4.0, places=8)
        self.assertAlmostEqual(float(phase), np.pi / 2.0, places=12)
        self.assertIn("elevation=90-theta", grid.history)
        self.assertEqual(grid.extra["source_format"], "CST wide theta/phi table")

    def test_wide_cst_rejects_corrupt_populated_rows(self):
        header = [
            "Frequency(GHz)", "Theta(deg)", "Phi(deg)",
            "RCS Theta-Theta(dBsm)", "Phase Theta-Theta(deg)",
        ]
        cases = (
            (["bad", 90.0, 0.0, 0.0, 0.0], "invalid frequency"),
            ([1.0, 90.0, 0.0, "bad", 0.0], "invalid rcs_vv_dbsm"),
            ([1.0, 90.0, 0.0, 0.0, "bad"], "invalid phase_vv_deg"),
        )
        for row, message in cases:
            with self.subTest(message=message):
                path = self._write([header, row])
                with self.assertRaisesRegex(ValueError, f"line 2: {message}"):
                    RcsGrid.read_CST(path)

    def test_wide_cst_rejects_ambiguous_field_and_unit_guesses(self):
        field_path = self._write(
            [[
                "Frequency(GHz)", "Theta(deg)", "Phi(deg)",
                "Abs(Theta) (V/m)", "Phase Theta(deg)",
            ], [1.0, 90.0, 0.0, 1.0, 0.0]]
        )
        headerless_path = self._write(
            [[1.0, 90.0, 0.0, 0.0, 0.0]]
        )
        unitless_path = self._write(
            [[
                "Frequency", "Theta(deg)", "Phi(deg)",
                "RCS Theta-Theta(dBsm)", "Phase Theta-Theta(deg)",
            ], [1.0, 90.0, 0.0, 0.0, 0.0]]
        )
        unitless_phase_path = self._write(
            [[
                "Frequency(GHz)", "Theta(deg)", "Phi(deg)",
                "RCS Theta-Theta(dBsm)", "Phase Theta-Theta",
            ], [1.0, 90.0, 0.0, 0.0, 0.0]]
        )
        radian_axes_path = self._write(
            [[
                "Frequency(GHz)", "Theta(rad)", "Phi(rad)",
                "RCS Theta-Theta(dBsm)", "Phase Theta-Theta(deg)",
            ], [1.0, np.pi / 2.0, 0.0, 0.0, 0.0]]
        )
        radian_phase_path = self._write(
            [[
                "Frequency(GHz)", "Theta(deg)", "Phi(deg)",
                "RCS Theta-Theta(dBsm)", "Phase Theta-Theta(rad)",
            ], [1.0, 90.0, 0.0, 0.0, 0.0]]
        )
        thz_path = self._write(
            [[
                "Frequency(THz)", "Theta(deg)", "Phi(deg)",
                "RCS Theta-Theta(dBsm)", "Phase Theta-Theta(deg)",
            ], [1.0, 90.0, 0.0, 0.0, 0.0]]
        )

        with self.assertRaisesRegex(ValueError, "explicit CST RCS header"):
            RcsGrid.read_CST(field_path)
        with self.assertRaisesRegex(ValueError, "explicit CST RCS header"):
            RcsGrid.read_CST(headerless_path)
        with self.assertRaisesRegex(ValueError, "must explicitly end"):
            RcsGrid.read_CST(unitless_path)
        with self.assertRaisesRegex(ValueError, "Ambiguous CST.*phases"):
            RcsGrid.read_CST(unitless_phase_path)
        with self.assertRaisesRegex(ValueError, "explicit CST RCS header"):
            RcsGrid.read_CST(radian_axes_path)
        with self.assertRaisesRegex(ValueError, "Ambiguous CST.*phases"):
            RcsGrid.read_CST(radian_phase_path)
        with self.assertRaisesRegex(ValueError, "must explicitly end"):
            RcsGrid.read_CST(thz_path)

    def test_wide_cst_preserves_established_cross_pol_component_mapping(self):
        path = self._write(
            [
                [
                    "Frequency(GHz)", "Theta(deg)", "Phi(deg)",
                    "RCS Theta-Theta(dBsm)", "RCS Phi-Theta(dBsm)",
                    "RCS Theta-Phi(dBsm)", "RCS Phi-Phi(dBsm)",
                    "Phase Theta-Theta(deg)", "Phase Phi-Theta(deg)",
                    "Phase Theta-Phi(deg)", "Phase Phi-Phi(deg)",
                ],
                [
                    3.0, 90.0, 0.0,
                    0.0, 3.010299957, 4.771212547, 6.020599913,
                    0.0, 10.0, 20.0, 30.0,
                ],
            ]
        )

        grid = RcsGrid.read_CST(path)

        np.testing.assert_array_equal(grid.polarizations, ["VV", "HV", "VH", "HH"])
        for polarization, expected_power, expected_phase in (
            ("VV", 1.0, 0.0),
            ("HV", 2.0, 10.0),
            ("VH", 3.0, 20.0),
            ("HH", 4.0, 30.0),
        ):
            power, phase = self._sample(grid, 0.0, 0.0, 3.0, polarization)
            self.assertAlmostEqual(float(power), expected_power, places=8)
            self.assertAlmostEqual(float(np.rad2deg(phase)), expected_phase, places=10)

    def test_wide_cst_compatibility_alias_and_duplicate_rejection(self):
        rows = [
            [
                "Frequency(GHz)", "Theta(deg)", "Phi(deg)",
                "RCS Theta-Theta(dBsm)", "Phase Theta-Theta(deg)",
            ],
            [1.0, 90.0, 180.0, 0.0, 0.0],
        ]
        path = self._write(rows)
        canonical = RcsGrid.read_CST(path)
        compatibility = RcsGrid.load_theta_phi_csv(path)
        np.testing.assert_allclose(compatibility.rcs_power, canonical.rcs_power)
        np.testing.assert_allclose(compatibility.rcs_phase, canonical.rcs_phase)
        np.testing.assert_allclose(compatibility.azimuths, canonical.azimuths)

        matching_seam_path = self._write(
            rows + [[1.0, 90.0, -180.0, 0.0, 0.0]]
        )
        matching_seam = RcsGrid.read_CST(matching_seam_path)
        np.testing.assert_allclose(matching_seam.azimuths, [-180.0])

        conflicting_path = self._write(
            rows + [[1.0, 90.0, -180.0, 3.0, 0.0]]
        )
        with self.assertRaisesRegex(
            ValueError, "conflicting duplicate CST theta/phi sample"
        ):
            RcsGrid.read_CST(conflicting_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
