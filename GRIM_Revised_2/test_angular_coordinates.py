"""Physics/metadata regressions for the deliberately narrow Conic/GC tool."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from grim_dataset import (
    GRIM_GC_CONVENTION,
    LEGACY_PTM_GC_CONVENTION,
    RcsGrid,
)


class EquatorialConicGcTests(unittest.TestCase):
    def _grid(
        self,
        *,
        coordinate_system="conic",
        azimuths=(0.0, 90.0, 180.0, 270.0),
        elevation=0.0,
        polarizations=("VV", "HH"),
        azimuth_unit="deg",
        elevation_unit="deg",
        convention=None,
        roll=0.0,
        tilt=0.0,
    ):
        azimuths = np.asarray(azimuths, dtype=float)
        polarizations = np.asarray(polarizations)
        shape = (azimuths.size, 1, 2, polarizations.size)
        real = np.arange(np.prod(shape), dtype=float).reshape(shape) + 1.0
        field = real + 1j * (0.25 * real)
        units = {
            "azimuth": azimuth_unit,
            "elevation": elevation_unit,
            "frequency": "GHz",
            "rcs_log_unit": "dBsm",
            "rcs_linear_quantity": "sigma_3d",
            "angular_coordinate_system": coordinate_system,
            "angular_roll_deg": roll,
            "angular_tilt_deg": tilt,
        }
        if convention is not None:
            units["great_circle_coordinate_convention"] = convention
        return RcsGrid(
            azimuths,
            [elevation],
            [9.0, 10.0],
            polarizations,
            rcs=field,
            history="source history",
            units=units,
            extra={
                "phase_reference": "exp(-jkr)",
                "rcs_amp_real": field.real.copy(),
                "rcs_amp_imag": field.imag.copy(),
                "scalar_note": "keep me",
                "solver_metadata_json": "stale grid binding",
                "production_mesh_certification_json": "stale certificate",
            },
        )

    def test_symmetric_exact_round_trip_reorders_field_and_raw_amplitude(self):
        source = self._grid()
        converted = source.convert_equatorial_conic_gc("conic_to_gc")

        self.assertEqual(converted.angular_coordinate_system(), "great_circle")
        self.assertEqual(
            converted.great_circle_coordinate_convention(), GRIM_GC_CONVENTION
        )
        np.testing.assert_array_equal(
            converted.azimuths, [-180.0, -90.0, 0.0, 90.0]
        )
        self.assertNotIn("solver_metadata_json", converted.extra)
        self.assertNotIn("production_mesh_certification_json", converted.extra)
        expected_order = [2, 3, 0, 1]
        np.testing.assert_allclose(converted.rcs, source.rcs[expected_order, ...])
        np.testing.assert_array_equal(
            converted.extra["rcs_amp_real"],
            source.extra["rcs_amp_real"][expected_order, ...],
        )
        self.assertIn("Conic->GC exact equatorial", converted.history)

        restored = converted.convert_equatorial_conic_gc("gc_to_conic")
        self.assertEqual(restored.angular_coordinate_system(), "conic")
        np.testing.assert_array_equal(restored.azimuths, converted.azimuths)
        np.testing.assert_allclose(restored.rcs, converted.rcs)
        self.assertEqual(restored.extra["phase_reference"], "exp(-jkr)")
        self.assertEqual(restored.extra["scalar_note"], "keep me")
        self.assertIn("GC->Conic exact equatorial", restored.history)

    def test_gc_to_conic_ptm_history_exports_as_ascii_pio_metadata(self):
        source = self._grid(
            coordinate_system="great_circle",
            azimuths=(-90.0, 0.0, 90.0),
            convention=GRIM_GC_CONVENTION,
        )
        converted = source.convert_equatorial_conic_gc("gc_to_conic")
        converted.history += (
            "\nGC→Conic relabel from PTM; Δ phase at 90°; "
            "Σ body ⊕ feature; coherent ÷ reference; uncommon marker Ω"
        )

        with tempfile.TemporaryDirectory() as folder:
            output = converted.save_pio(
                os.path.join(folder, "converted → conic.pio"),
                pol_idx=0,
                precision="double",
            )
            with open(output, "rb") as stream:
                header = stream.read().split(b"Offset=", 1)[0]
            restored = RcsGrid.load_pio(output)

        header_text = header.decode("ascii")
        self.assertIn("Name=converted -> conic", header_text)
        self.assertIn("GC->Conic relabel from PTM", header_text)
        self.assertIn("Delta phase at 90 deg", header_text)
        self.assertIn("Sum body + feature; coherent / reference", header_text)
        self.assertIn(r"uncommon marker \u03a9", header_text)
        np.testing.assert_allclose(restored.rcs, converted.rcs[..., 0:1])

    def test_radian_axes_are_wrapped_in_radians(self):
        source = self._grid(
            azimuths=(0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0),
            azimuth_unit="rad",
            elevation_unit="rad",
        )
        converted = source.convert_equatorial_conic_gc("conic_to_gc")
        np.testing.assert_allclose(
            converted.azimuths, [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0]
        )
        self.assertEqual(converted.units["azimuth"], "rad")

    def test_unmarked_legacy_ptm_requires_explicit_attestation(self):
        source = self._grid(
            coordinate_system="great_circle",
            azimuths=(-90.0, 0.0, 90.0),
            convention=LEGACY_PTM_GC_CONVENTION,
        )
        with self.assertRaisesRegex(ValueError, "aspect sign/origin"):
            source.convert_equatorial_conic_gc("gc_to_conic")

        converted = source.convert_equatorial_conic_gc(
            "gc_to_conic", attest_legacy_ptm_convention=True
        )
        self.assertEqual(converted.angular_coordinate_system(), "conic")
        self.assertIn("user-attested legacy PTM", converted.history)

        other = self._grid(
            coordinate_system="great_circle",
            azimuths=(-90.0, 0.0, 90.0),
            convention="other_gc_v2",
        )
        with self.assertRaisesRegex(ValueError, "unsupported great-circle"):
            other.convert_equatorial_conic_gc(
                "gc_to_conic", attest_legacy_ptm_convention=True
            )

    def test_known_and_unmarked_gc_grids_are_not_physically_interchangeable(self):
        known = self._grid().convert_equatorial_conic_gc("conic_to_gc")
        units = dict(known.units)
        units["great_circle_coordinate_convention"] = LEGACY_PTM_GC_CONVENTION
        unmarked = RcsGrid(
            known.azimuths,
            known.elevations,
            known.frequencies,
            known.polarizations,
            rcs_power=known.rcs_power,
            rcs_phase=known.rcs_phase,
            units=units,
        )
        with self.assertRaisesRegex(ValueError, "coordinate convention mismatch"):
            known.coherent_add(unmarked)

        with self.assertRaisesRegex(ValueError, "coordinate convention mismatch"):
            RcsGrid.join_many(known, unmarked)

        conic = self._grid(azimuths=(300.0, 330.0))
        with self.assertRaisesRegex(ValueError, "angular coordinate system mismatch"):
            RcsGrid.join_many(known, conic)

    def test_extra_only_gc_metadata_is_canonicalized_and_survives_transform(self):
        field = np.ones((2, 1, 2, 1), dtype=np.complex64)
        source = RcsGrid(
            [-10.0, 10.0],
            [0.0],
            [9.0, 10.0],
            ["VV"],
            rcs=field,
            units={"frequency": "GHz"},
            extra={
                "angular_coordinate_system": "great_circle",
                "great_circle_coordinate_convention": GRIM_GC_CONVENTION,
                "ptm_roll": 12.5,
                "ptm_tilt": -1.0,
            },
        )
        self.assertEqual(source.units["angular_coordinate_system"], "great_circle")
        self.assertEqual(
            source.units["great_circle_coordinate_convention"], GRIM_GC_CONVENTION
        )
        transformed = source.statistics_dataset(
            statistic="mean",
            axes=["frequency"],
            domain="magnitude",
            broadcast_reduced=True,
        )
        self.assertEqual(transformed.angular_coordinate_system(), "great_circle")
        self.assertEqual(
            transformed.great_circle_coordinate_convention(), GRIM_GC_CONVENTION
        )
        self.assertEqual(
            transformed.angular_frame_orientation_deg(), (12.5, -1.0)
        )

    def test_grim_file_round_trip_preserves_gc_convention(self):
        source = self._grid().convert_equatorial_conic_gc("conic_to_gc")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = source.save(os.path.join(temp_dir, "known_gc.grim"))
            loaded = RcsGrid.load(path)
        self.assertEqual(loaded.angular_coordinate_system(), "great_circle")
        self.assertEqual(
            loaded.great_circle_coordinate_convention(), GRIM_GC_CONVENTION
        )

    def test_rejects_wrong_tag_nonzero_cut_orientation_crosspol_and_seam(self):
        source = self._grid()
        with self.assertRaisesRegex(ValueError, "requires a source tagged great_circle"):
            source.convert_equatorial_conic_gc("gc_to_conic")

        unknown = self._grid(coordinate_system="wedge_turntable")
        with self.assertRaisesRegex(ValueError, "does not support"):
            unknown.convert_equatorial_conic_gc("conic_to_gc")

        nonzero = self._grid(elevation=1.0)
        with self.assertRaisesRegex(ValueError, "0 degree elevation"):
            nonzero.convert_equatorial_conic_gc("conic_to_gc")

        oriented = self._grid(roll=1.0)
        with self.assertRaisesRegex(ValueError, "roll=tilt=0"):
            oriented.convert_equatorial_conic_gc("conic_to_gc")

        crosspol = self._grid(polarizations=("VH", "HV"))
        with self.assertRaisesRegex(ValueError, "VV/HH only"):
            crosspol.convert_equatorial_conic_gc("conic_to_gc")

        seam = self._grid(azimuths=(-180.0, 0.0, 180.0))
        with self.assertRaisesRegex(ValueError, "seam-alias"):
            seam.convert_equatorial_conic_gc("conic_to_gc")

    def test_grim_written_ptm_marker_makes_own_file_safe_to_relabel(self):
        azimuths = np.asarray([-90.0, 0.0, 90.0])
        frequencies = np.linspace(8.0, 9.0, 37)
        field = np.ones((3, 1, 37, 1), dtype=np.complex64) * (2.0 + 0.5j)
        source = RcsGrid(
            azimuths,
            [0.0],
            frequencies,
            ["VV"],
            rcs=field,
            units={
                "azimuth": "deg",
                "elevation": "deg",
                "frequency": "GHz",
                "rcs_log_unit": "dBsm",
                "rcs_linear_quantity": "sigma_3d",
                "angular_coordinate_system": "conic",
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "grim_equator.ptm")
            source.save_ptm(path)
            loaded = RcsGrid.load_ptm(path)

        self.assertEqual(
            loaded.great_circle_coordinate_convention(), GRIM_GC_CONVENTION
        )
        converted = loaded.convert_equatorial_conic_gc("gc_to_conic")
        np.testing.assert_allclose(converted.rcs, loaded.rcs)


if __name__ == "__main__":
    unittest.main()
