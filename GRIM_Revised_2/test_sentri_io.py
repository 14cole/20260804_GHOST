"""Format and dispatch regressions for CREATE-RF SENTRi RCS tables."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from grim_dataset import RcsGrid
from grim_headless import load_dataset, read_SENTRi


COMPACT_HEADER = (
    "freq_MHz_,theta_deg_,phi_deg_,rcs_pp_dBsm_,"
    "efield_phase_pp_deg_,rcs_tt_dBsm_,efield_phase_tt_deg_,"
    "rcs_pt_dBsm_,efield_phase_pt_deg_,rcs_tp_dBsm_,"
    "efield_phase_tp_deg_"
)

COMPACT_UNITS = (
    "MHz,deg,deg,dBsm,deg,dBsm,deg,dBsm,deg,dBsm,deg"
)

DESCRIPTIVE_HEADER = (
    "Frequency,Theta,Phi,RCSPhiScat_PhiInc,PhasePhi_Phi,"
    "RCSThetaScat_ThetaInc,PhaseTheta_Theta,"
    "RCSPhiScat_ThetaInc,PhasePhi_Theta,"
    "RCSThetaScat_PhiInc,PhaseTheta_Phi"
)

DESCRIPTIVE_UNITS = (
    "Hz,deg,deg,dBsm,deg,dBsm,deg,dBsm,deg,dBsm,deg"
)


class SentriReaderTest(unittest.TestCase):
    def _write(self, suffix: str, text: str, *, bom: bool = False) -> str:
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        handle.close()
        encoding = "utf-8-sig" if bom else "utf-8"
        with open(handle.name, "w", encoding=encoding, newline="") as stream:
            stream.write(text)
        self.addCleanup(lambda: os.path.exists(handle.name) and os.unlink(handle.name))
        return handle.name

    def test_compact_schema_axes_polarizations_power_and_phase(self) -> None:
        path = self._write(
            ".csv",
            COMPACT_HEADER
            + "\n"
            + COMPACT_UNITS
            + "\n1000,80,190,0,90,-20,-45,-6.020599913,180,"
            + "6.020599913,0\n",
            bom=True,
        )
        grid = RcsGrid.read_SENTRi(path)

        np.testing.assert_allclose(grid.frequencies, [1.0])
        np.testing.assert_allclose(grid.elevations, [80.0])
        np.testing.assert_allclose(grid.azimuths, [-170.0])
        self.assertEqual(grid.polarizations.tolist(), ["VV", "HV", "VH", "HH"])

        expected_power = np.asarray([0.01, 0.25, 4.0, 1.0])
        np.testing.assert_allclose(grid.rcs_power[0, 0, 0, :], expected_power)
        expected_phase = np.deg2rad([-45.0, 180.0, 0.0, 90.0])
        np.testing.assert_allclose(grid.rcs_phase[0, 0, 0, :], expected_phase)
        np.testing.assert_allclose(
            grid.rcs[0, 0, 0, :],
            np.sqrt(expected_power) * np.exp(1j * expected_phase),
        )
        self.assertIn("stored phase=reported", grid.history)
        self.assertEqual(
            grid.extra["sentri_coordinate_mapping"],
            "elevation=theta; azimuth=wrapped phi",
        )
        self.assertIn("exp(+j*deg2rad", grid.extra["sentri_phase_mapping"])
        self.assertTrue(grid.extra["sentri_units_row_present"])

    def test_descriptive_schema_and_tab_delimited_dispatch(self) -> None:
        header = DESCRIPTIVE_HEADER.replace(",", "\t")
        units = DESCRIPTIVE_UNITS.replace(",", "\t")
        row = "1000000000\t100\t10\t0\t10\t-10\t20\t-20\t30\t-30\t40\n"
        path = self._write(".txt", header + "\n" + units + "\n" + row)

        direct = read_SENTRi(path)
        dropped = load_dataset(path)
        np.testing.assert_allclose(direct.elevations, [100.0])
        np.testing.assert_allclose(direct.frequencies, [1.0])
        np.testing.assert_allclose(direct.rcs_power, dropped.rcs_power)
        np.testing.assert_allclose(direct.rcs_phase, dropped.rcs_phase)
        np.testing.assert_allclose(
            direct.rcs_phase[0, 0, 0, :],
            np.deg2rad([20.0, 30.0, 40.0, 10.0]),
        )
        np.testing.assert_allclose(
            direct.rcs_power[0, 0, 0, :],
            10.0 ** (np.asarray([-10.0, -20.0, -30.0, 0.0]) / 10.0),
        )
        self.assertTrue(direct.extra["sentri_units_row_present"])

    def test_units_row_is_validated_instead_of_treated_as_data(self) -> None:
        wrong_units = self._write(
            ".csv",
            DESCRIPTIVE_HEADER
            + "\nGHz,deg,deg,dBsm,deg,dBsm,deg,dBsm,deg,dBsm,deg"
            + "\n1,90,0,0,0,0,0,0,0,0,0\n",
        )
        with self.assertRaisesRegex(ValueError, "invalid SENTRi units row"):
            load_dataset(wrong_units)

        wrong_angle = self._write(
            ".csv",
            DESCRIPTIVE_HEADER
            + "\nHz,rad,deg,dBsm,deg,dBsm,deg,dBsm,deg,dBsm,deg"
            + "\n1000000000,1,0,0,0,0,0,0,0,0,0\n",
        )
        with self.assertRaisesRegex(ValueError, "theta='rad'.*expected deg"):
            load_dataset(wrong_angle)

        truncated_units = self._write(
            ".csv",
            DESCRIPTIVE_HEADER
            + "\nHz,deg,deg"
            + "\n1000000000,90,0,0,0,0,0,0,0,0,0\n",
        )
        with self.assertRaisesRegex(ValueError, "invalid SENTRi units row"):
            load_dataset(truncated_units)

        no_units = self._write(
            ".csv",
            DESCRIPTIVE_HEADER
            + "\n1000000000,90,0,0,0,0,0,0,0,0,0\n",
        )
        legacy_grid = load_dataset(no_units)
        self.assertFalse(legacy_grid.extra["sentri_units_row_present"])

        bad_data_after_units = self._write(
            ".csv",
            DESCRIPTIVE_HEADER
            + "\n"
            + DESCRIPTIVE_UNITS
            + "\n1000000000,bad,0,0,0,0,0,0,0,0,0\n",
        )
        with self.assertRaisesRegex(ValueError, "line 3: invalid theta"):
            load_dataset(bad_data_after_units)

    def test_csv_drop_dispatch_prefers_strict_sentri_signature(self) -> None:
        path = self._write(
            ".csv",
            DESCRIPTIVE_HEADER
            + "\n2000000000,90,0,3,0,2,0,1,0,0,0\n",
        )
        grid = load_dataset(path)
        self.assertTrue(str(grid.extra["source_format"]).startswith("SENTRi"))
        np.testing.assert_allclose(grid.frequencies, [2.0])

    def test_positive_azimuth_is_not_negated(self) -> None:
        path = self._write(
            ".csv",
            COMPACT_HEADER
            + "\n1000,90,90,0,0,0,0,0,0,0,0\n",
        )
        grid = load_dataset(path)
        np.testing.assert_allclose(grid.azimuths, [90.0])

    def test_signed_seam_always_uses_positive_180_degree_record(self) -> None:
        row = "1000,90,{phi},0,0,0,0,0,0,0,0"
        matching = self._write(
            ".csv",
            COMPACT_HEADER
            + "\n"
            + row.format(phi=-180)
            + "\n"
            + row.format(phi=180)
            + "\n",
        )
        grid = RcsGrid.read_SENTRi(matching)
        np.testing.assert_allclose(grid.azimuths, [-180.0])

        row_negative = "1000,90,-180,1,21,2,22,3,23,4,24"
        row_positive = "1000,90,180,11,31,12,32,13,33,14,34"
        for rows in ((row_negative, row_positive), (row_positive, row_negative)):
            with self.subTest(order=rows):
                path = self._write(
                    ".csv", COMPACT_HEADER + "\n" + "\n".join(rows) + "\n"
                )
                signed_grid = RcsGrid.read_SENTRi(path)

                np.testing.assert_allclose(signed_grid.azimuths, [-180.0])
                np.testing.assert_allclose(
                    signed_grid.rcs_power[0, 0, 0, :],
                    10.0 ** (np.asarray([12.0, 13.0, 14.0, 11.0]) / 10.0),
                )
                np.testing.assert_allclose(
                    signed_grid.rcs_phase[0, 0, 0, :],
                    np.deg2rad([32.0, 33.0, 34.0, 31.0]),
                )
                self.assertTrue(
                    signed_grid.extra["sentri_signed_180_precedence_used"]
                )
                self.assertIn(
                    "phi=+180 supplies canonical azimuth -180",
                    signed_grid.extra["sentri_signed_180_seam_policy"],
                )

        # Repeated rows at the same source angle are not a closed-sweep seam
        # pair.  Conflicting values must still fail instead of being hidden.
        conflicting_duplicate = self._write(
            ".csv",
            COMPACT_HEADER
            + "\n"
            + row.format(phi=-180)
            + "\n1000,90,-180,1,0,0,0,0,0,0,0\n",
        )
        with self.assertRaisesRegex(ValueError, "conflicting duplicate SENTRi"):
            RcsGrid.read_SENTRi(conflicting_duplicate)

        # A closing endpoint must not mask a conflicting repeat at the other
        # source endpoint.  The failure is independent of where +180 appears.
        positive = "1000,90,180,11,31,12,32,13,33,14,34"
        negative_a = "1000,90,-180,1,21,2,22,3,23,4,24"
        negative_b = "1000,90,-180,5,25,6,26,7,27,8,28"
        for rows in (
            (positive, negative_a, negative_b),
            (negative_a, positive, negative_b),
            (negative_a, negative_b, positive),
        ):
            with self.subTest(conflicting_triplet=rows):
                path = self._write(
                    ".csv", COMPACT_HEADER + "\n" + "\n".join(rows) + "\n"
                )
                with self.assertRaisesRegex(
                    ValueError, "conflicting duplicate SENTRi"
                ):
                    RcsGrid.read_SENTRi(path)

        # Equivalent repeats remain harmless even when the closing endpoint
        # appears between them; +180 still supplies the canonical sample.
        equivalent_repeat = self._write(
            ".csv",
            COMPACT_HEADER
            + "\n"
            + "\n".join((negative_a, positive, negative_a))
            + "\n",
        )
        repeated_grid = RcsGrid.read_SENTRi(equivalent_repeat)
        np.testing.assert_allclose(
            repeated_grid.rcs_power[0, 0, 0, :],
            10.0 ** (np.asarray([12.0, 13.0, 14.0, 11.0]) / 10.0),
        )

        # A zero-amplitude sample has no physical phase.  SENTRi can report
        # different arbitrary phase values at the equivalent -180/+180 seam;
        # they must collapse to the same zero complex sample.
        zero_seam = self._write(
            ".csv",
            COMPACT_HEADER
            + "\n1000,90,-180,-Inf,0,-Inf,10,-Inf,20,-Inf,30"
            + "\n1000,90,180,-Inf,40,-Inf,50,-Inf,60,-Inf,70\n",
        )
        zero_grid = RcsGrid.read_SENTRi(zero_seam)
        np.testing.assert_allclose(zero_grid.azimuths, [-180.0])
        np.testing.assert_allclose(zero_grid.rcs_power, 0.0)

    def test_csv_drop_accepts_conflicting_signed_closed_sweep_endpoints(self) -> None:
        path = self._write(
            ".csv",
            DESCRIPTIVE_HEADER
            + "\n"
            + DESCRIPTIVE_UNITS
            + "\n1000000000,90,-180,1,21,2,22,3,23,4,24"
            + "\n1000000000,90,180,11,31,12,32,13,33,14,34\n",
        )

        grid = load_dataset(path)

        np.testing.assert_allclose(grid.azimuths, [-180.0])
        np.testing.assert_allclose(
            grid.rcs_power[0, 0, 0, :],
            10.0 ** (np.asarray([12.0, 13.0, 14.0, 11.0]) / 10.0),
        )
        np.testing.assert_allclose(
            grid.rcs_phase[0, 0, 0, :],
            np.deg2rad([32.0, 33.0, 34.0, 31.0]),
        )
        self.assertTrue(grid.extra["sentri_signed_180_precedence_used"])

    def test_zero_360_seam_always_uses_the_360_degree_record(self) -> None:
        row_zero = "1000,90,0,1,21,2,22,3,23,4,24"
        row_360 = "1000,90,360,11,31,12,32,13,33,14,34"

        for rows in ((row_zero, row_360), (row_360, row_zero)):
            with self.subTest(order=rows):
                path = self._write(
                    ".csv", COMPACT_HEADER + "\n" + "\n".join(rows) + "\n"
                )
                grid = RcsGrid.read_SENTRi(path)

                np.testing.assert_allclose(grid.azimuths, [0.0])
                np.testing.assert_allclose(
                    grid.rcs_power[0, 0, 0, :],
                    10.0 ** (np.asarray([12.0, 13.0, 14.0, 11.0]) / 10.0),
                )
                np.testing.assert_allclose(
                    grid.rcs_phase[0, 0, 0, :],
                    np.deg2rad([32.0, 33.0, 34.0, 31.0]),
                )
                self.assertTrue(
                    grid.extra["sentri_zero_360_precedence_used"]
                )
                self.assertIn(
                    "phi=360 supplies canonical azimuth 0",
                    grid.extra["sentri_zero_360_seam_policy"],
                )

    def test_sentri_endpoint_roundoff_is_normalized_before_seam_precedence(self):
        row_zero = "1000,-0.0000000005,0,1,21,2,22,3,23,4,24"
        row_360 = "1000,0,360.0000000005,11,31,12,32,13,33,14,34"
        path = self._write(
            ".csv", COMPACT_HEADER + "\n" + row_zero + "\n" + row_360 + "\n"
        )

        grid = RcsGrid.read_SENTRi(path)

        np.testing.assert_array_equal(grid.azimuths, [0.0])
        np.testing.assert_array_equal(grid.elevations, [0.0])
        np.testing.assert_allclose(
            grid.rcs_power[0, 0, 0, :],
            10.0 ** (np.asarray([12.0, 13.0, 14.0, 11.0]) / 10.0),
        )
        self.assertTrue(grid.extra["sentri_zero_360_precedence_used"])
        np.testing.assert_array_equal(
            grid.convert_sentri_elevation_to_grim().elevations,
            [90.0],
        )

        for negative_phi in (-180.0000000005, -179.9999999995):
            for positive_phi in (179.9999999995, 180.0000000005):
                negative_row = (
                    f"1000,90,{negative_phi:.10f},0,0,0,0,0,0,0,0"
                )
                positive_row = (
                    f"1000,90,{positive_phi:.10f},1,0,0,0,0,0,0,0"
                )
                for rows in (
                    (negative_row, positive_row),
                    (positive_row, negative_row),
                ):
                    with self.subTest(
                        negative_phi=negative_phi,
                        positive_phi=positive_phi,
                        order=rows,
                    ):
                        fuzzy_signed_seam = self._write(
                            ".csv",
                            COMPACT_HEADER + "\n" + "\n".join(rows) + "\n",
                        )
                        fuzzy_grid = RcsGrid.read_SENTRi(fuzzy_signed_seam)
                        np.testing.assert_array_equal(
                            fuzzy_grid.azimuths, [-180.0]
                        )
                        np.testing.assert_allclose(
                            fuzzy_grid.rcs_power[0, 0, 0, :],
                            [1.0, 1.0, 1.0, 10.0 ** 0.1],
                        )
                        self.assertTrue(
                            fuzzy_grid.extra[
                                "sentri_signed_180_precedence_used"
                            ]
                        )

    def test_explicit_sentri_elevation_conversion_is_sorted_and_lossless(self) -> None:
        rows = (
            "1000,0,10,1,11,2,12,3,13,4,14",
            "1000,90,10,5,15,6,16,7,17,8,18",
            "1000,180,10,9,19,10,20,11,21,12,22",
        )
        path = self._write(
            ".csv", COMPACT_HEADER + "\n" + "\n".join(rows) + "\n"
        )
        grid = RcsGrid.read_SENTRi(path)
        aligned = np.arange(grid.rcs_power.size, dtype=float).reshape(
            grid.rcs_power.shape
        )
        grid.extra["aligned"] = aligned
        grid.extra["solver_metadata_json"] = "stale"

        converted = grid.convert_sentri_elevation_to_grim()

        np.testing.assert_allclose(grid.elevations, [0.0, 90.0, 180.0])
        np.testing.assert_allclose(converted.elevations, [-90.0, 0.0, 90.0])
        np.testing.assert_allclose(
            converted.rcs_power,
            np.take(grid.rcs_power, [2, 1, 0], axis=1),
        )
        np.testing.assert_allclose(
            converted.rcs_phase,
            np.take(grid.rcs_phase, [2, 1, 0], axis=1),
        )
        np.testing.assert_array_equal(
            converted.extra["aligned"], np.take(aligned, [2, 1, 0], axis=1)
        )
        self.assertNotIn("solver_metadata_json", converted.extra)
        self.assertEqual(
            converted.units["elevation_coordinate_convention"],
            "grim_elevation_waterline_zero_top_positive",
        )
        self.assertEqual(
            converted.extra["assembly_angular_coordinate_contract"],
            "ghost.radar-azimuth-elevation.coming-from.deg.v1",
        )
        self.assertIn("elevation=90-theta", converted.history)
        self.assertIn("no interpolation or phase change", converted.history)
        with self.assertRaisesRegex(ValueError, "already uses GRIM"):
            converted.convert_sentri_elevation_to_grim()

        generic = RcsGrid(
            [0.0],
            [0.0],
            [1.0],
            ["VV"],
            rcs=np.ones((1, 1, 1, 1), dtype=np.complex128),
            units={"elevation": "deg", "frequency": "GHz"},
        )
        with self.assertRaisesRegex(ValueError, "not marked as native SENTRi"):
            generic.convert_sentri_elevation_to_grim()

        nearly_duplicate = RcsGrid(
            [0.0],
            [90.0, 90.0 + 5.0e-10],
            [1.0],
            ["VV"],
            rcs=np.ones((1, 2, 1, 1), dtype=np.complex128),
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "elevation_coordinate_convention": "sentri_theta_top_zero",
            },
            extra={"source_format": "SENTRi test table"},
        )
        with self.assertRaisesRegex(ValueError, "near-duplicate GRIM elevation"):
            nearly_duplicate.convert_sentri_elevation_to_grim()

    def test_incomplete_or_generic_theta_phi_table_is_not_sentri(self) -> None:
        path = self._write(
            ".csv",
            "Frequency,Theta,Phi,RCSPhiScat_PhiInc\n1000000000,90,0,0\n",
        )
        with self.assertRaisesRegex(ValueError, "complete SENTRi RCS header"):
            RcsGrid.read_SENTRi(path)

    def test_signed_vendor_header_commits_dispatch_to_sentri(self) -> None:
        malformed = self._write(
            ".txt",
            COMPACT_HEADER.replace(",", "\t")
            + "\n1000\t200\t0\t0\t0\t0\t0\t0\t0\t0\t0\n",
        )
        self.assertTrue(RcsGrid.has_SENTRi_signature(malformed))
        with self.assertRaisesRegex(ValueError, "theta must be in"):
            load_dataset(malformed)

        partial = self._write(
            ".txt",
            "freq_MHz_,theta_deg_,phi_deg_,rcs_pp_dBsm_,rcs_tt_dBsm_\n"
            "1000,90,0,0,0\n",
        )
        self.assertTrue(RcsGrid.has_SENTRi_signature(partial))
        with self.assertRaisesRegex(ValueError, "complete SENTRi RCS header"):
            load_dataset(partial)

    def test_dispatch_does_not_steal_existing_delimited_formats(self) -> None:
        native = self._write(
            ".csv",
            "azimuth,elevation,frequency,frequency_unit,polarization,"
            "rcs_log_unit,magnitude_linear,phase_deg\n"
            "0,0,3,GHz,VV,dBsm,2,0\n",
        )
        native_grid = load_dataset(native)
        np.testing.assert_allclose(native_grid.rcs_power, 2.0)
        self.assertIn("Loaded flat CSV", native_grid.history)

        wide_cst = self._write(
            ".csv",
            "Frequency(GHz),Theta(deg),Phi(deg),"
            "RCS Theta-Theta(dBsm),Phase Theta-Theta(deg)\n"
            "2,90,0,0,0\n",
        )
        cst_grid = load_dataset(wide_cst)
        self.assertEqual(cst_grid.extra["source_format"], "CST wide theta/phi table")

        legacy_txt = self._write(
            ".txt",
            "theta(deg) phi(deg) abs(rcs)(dbm^2) abs(theta)(dbm^2) "
            "phase(theta)(deg) abs(phi)(dbm^2) phase(phi)(deg) ax.ratio(db)\n"
            "0 0 0 0 0 0 0 0\n",
        )
        legacy_grid = load_dataset(legacy_txt)
        self.assertIn("Loaded theta/phi TXT", legacy_grid.history)

    def test_sparse_cartesian_cells_remain_nan(self) -> None:
        row1 = "1000,80,0,0,0,0,0,0,0,0,0"
        row2 = "2000,100,10,0,0,0,0,0,0,0,0"
        path = self._write(".csv", COMPACT_HEADER + "\n" + row1 + "\n" + row2 + "\n")
        grid = RcsGrid.read_SENTRi(path)
        self.assertEqual(grid.rcs_power.shape, (2, 2, 2, 4))
        self.assertGreater(int(np.count_nonzero(np.isnan(grid.rcs_power))), 0)


if __name__ == "__main__":
    unittest.main()
