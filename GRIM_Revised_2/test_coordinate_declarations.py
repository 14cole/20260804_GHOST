"""User coordinate declarations override format assumptions without moving data."""

import json
import os
import tempfile
import unittest

import numpy as np

from grim_dataset import GRIM_GC_CONVENTION, LEGACY_PTM_GC_CONVENTION, RcsGrid
from grim_headless import load_dataset
from grim_python import plot_datasets
from plot_modes.common import validate_plot_datasets
from test_ptm import _independent_fixture


class CoordinateDeclarationTests(unittest.TestCase):
    def test_matching_pio_ptm_nonzero_cut_can_overlay_after_declaration(self):
        with tempfile.TemporaryDirectory() as folder:
            ptm_path = os.path.join(folder, "same.ptm")
            _independent_fixture(ptm_path, start_aspect=-2.0, aspect_increment=1.0)
            ptm = load_dataset(ptm_path)
            # Encode the same numeric data as a PIO azimuth/elevation cut.
            source = RcsGrid(
                ptm.azimuths, ptm.elevations, ptm.frequencies, ptm.polarizations,
                rcs=ptm.rcs, units={"frequency": "GHz"},
            )
            pio = load_dataset(source.save_pio(os.path.join(folder, "same.pio")))
            self.assertEqual(ptm.elevations.tolist(), [7.5])
            with self.assertRaisesRegex(ValueError, "Set Coordinates"):
                validate_plot_datasets([("PIO", pio), ("PTM", ptm)], phase=False, linear=False)

            corrected = ptm.set_angular_coordinate_system("conic")
            for attr in ("azimuths", "elevations", "frequencies", "polarizations",
                         "rcs_power", "rcs_phase"):
                np.testing.assert_array_equal(getattr(corrected, attr), getattr(ptm, attr))
            self.assertEqual(corrected.angular_frame_orientation_deg(), (0.0, 0.0))
            self.assertEqual(ptm.angular_coordinate_system(), "great_circle")
            self.assertEqual(ptm.angular_frame_orientation_deg(), (1.25, -2.5))
            figure = plot_datasets(
                [("PIO", pio), ("PTM", corrected)], mode="azimuth_rect",
                azimuths=pio.azimuths, elevations=pio.elevations,
                frequencies=pio.frequencies[:1], polarization="HH",
            )
            axis = figure.axes[0]
            self.assertEqual(axis.get_xlabel(), "Azimuth (deg)")
            self.assertEqual(len(axis.lines), 2)
            np.testing.assert_allclose(axis.lines[0].get_xdata(), axis.lines[1].get_xdata())
            np.testing.assert_allclose(axis.lines[0].get_ydata(), axis.lines[1].get_ydata(), atol=1e-5)

            restored = RcsGrid.load(corrected.save(os.path.join(folder, "corrected.grim")))
            self.assertEqual(restored.angular_coordinate_system(), "conic")
            self.assertIn("User declared coordinates", restored.history)
            declaration = json.loads(str(
                np.asarray(restored.extra["angular_coordinate_declaration_json"]).item()
            ))
            self.assertEqual(declaration["source_system"], "great_circle")
            self.assertFalse(declaration["numeric_data_changed"])

    def test_gc_declaration_preserves_all_samples_and_sets_shared_frame(self):
        shape = (3, 2, 2, 4)
        power = np.arange(np.prod(shape), dtype=float).reshape(shape)
        phase = power / 10.0
        phase.flat[4] = np.nan
        source = RcsGrid(
            [-180, 0, 180], [5, 10], [9, 10], ["VV", "HV", "VH", "HH"],
            rcs_power=power, rcs_phase=phase,
            extra={"aligned": power.copy(), "solver_metadata_json": "old",
                   "assembly_angular_coordinate_contract": "old"},
        )
        corrected = source.set_angular_coordinate_system(
            "great_circle", gc_convention=GRIM_GC_CONVENTION,
            roll_deg=1.25, tilt_deg=-2.5,
        )
        np.testing.assert_array_equal(corrected.rcs_power, source.rcs_power)
        np.testing.assert_array_equal(corrected.rcs_phase, source.rcs_phase)
        np.testing.assert_array_equal(corrected.azimuths, [-180, 0, 180])
        np.testing.assert_array_equal(corrected.extra["aligned"], power)
        self.assertEqual(corrected.angular_frame_orientation_deg(), (1.25, -2.5))
        self.assertEqual(corrected.great_circle_coordinate_convention(), GRIM_GC_CONVENTION)
        self.assertEqual(corrected.extra["angular_coordinate_system"], "great_circle")
        self.assertNotIn("solver_metadata_json", corrected.extra)
        self.assertNotIn("assembly_angular_coordinate_contract", corrected.extra)
        corrected.rcs_power.flat[0] = 999
        corrected.extra["aligned"].flat[0] = 999
        self.assertEqual(source.rcs_power.flat[0], 0)
        self.assertEqual(source.extra["aligned"].flat[0], 0)

        legacy = source.set_angular_coordinate_system("great_circle")
        self.assertEqual(legacy.great_circle_coordinate_convention(), LEGACY_PTM_GC_CONVENTION)
        matching = legacy.set_angular_coordinate_system(
            "great_circle", gc_convention=GRIM_GC_CONVENTION,
            roll_deg=1.25, tilt_deg=-2.5,
        )
        validate_plot_datasets([("first", corrected), ("second", matching)], phase=False, linear=False)

    def test_rejects_invalid_declarations_before_copying(self):
        grid = RcsGrid([0], [0], [1], ["VV"], rcs=np.ones((1, 1, 1, 1)))
        for system in (None, "", "wedge", "typo"):
            with self.subTest(system=system), self.assertRaisesRegex(ValueError, "coordinate_system"):
                grid.set_angular_coordinate_system(system)
        with self.assertRaisesRegex(ValueError, "convention"):
            grid.set_angular_coordinate_system("great_circle", gc_convention="typo")
        with self.assertRaisesRegex(ValueError, "finite"):
            grid.set_angular_coordinate_system("great_circle", roll_deg=np.nan)


if __name__ == "__main__":
    unittest.main()
