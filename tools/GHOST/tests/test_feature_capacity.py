"""Production admission checks for vehicle-scale Feature Assembly jobs."""

from __future__ import annotations

import errno
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


BACKEND = Path(__file__).resolve().parents[1] / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import components  # noqa: E402
import feature_sum  # noqa: E402
from occluder import Occluder  # noqa: E402


SMALL_GRID = {
    "frequencies_ghz": [1.0],
    "azimuths_deg": [0.0, 45.0],
    "elevations_deg": [0.0],
    "axis_az_deg": 0.0,
    "axis_el_deg": 0.0,
    "roll_deg": 0.0,
}


def _write_empty_base(path: Path) -> None:
    feature_sum.export_radar_grim(
        str(path), bor_result=None, placements=[], **SMALL_GRID
    )


class FeatureCapacityTests(unittest.TestCase):
    def test_estimate_is_monotone_in_grid_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.grim"
            base.write_bytes(b"small lower-bound archive")
            small = feature_sum.estimate_feature_assembly_capacity(
                str(base), str(root / "small.grim"), radar_grid=SMALL_GRID
            )
            larger_grid = dict(SMALL_GRID)
            larger_grid["frequencies_ghz"] = [1.0, 2.0, 3.0]
            larger_grid["elevations_deg"] = [-10.0, 0.0, 10.0]
            large = feature_sum.estimate_feature_assembly_capacity(
                str(base), str(root / "large.grim"), radar_grid=larger_grid
            )

            self.assertGreater(large.grid_cells, small.grid_cells)
            self.assertGreater(
                large.estimated_peak_memory_bytes,
                small.estimated_peak_memory_bytes,
            )
            self.assertGreater(
                large.estimated_scratch_bytes, small.estimated_scratch_bytes
            )

    def test_shadow_estimate_uses_packed_point_and_line_masks(self):
        triangles = np.asarray([[[0.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0]]])
        occluder = Occluder(triangles, bias=0.0)
        grid = dict(SMALL_GRID)
        grid["azimuths_deg"] = list(range(9))
        placements = [{
            # Direct API fixed grids need not supply pre-registered shadow
            # points; four solver pieces are derived from this geometry.
            "perimeter": np.asarray([[[0.0, 0.0, 0.0],
                                        [1.0, 0.0, 0.0]]]),
            "max_piece_length_m": 0.25,
        }]
        points = [{"location": np.zeros(3)}, {"location": np.ones(3)}]
        estimate = feature_sum.estimate_feature_assembly_capacity(
            "missing.grim", "output.grim", radar_grid=grid,
            placements=placements, points=points, occluder=occluder,
        )

        # Nine looks occupy two packed bytes per feature/piece: 2*(2+4).
        self.assertEqual(estimate.shadow_mask_bytes, 12)

    def test_memory_rejection_is_actionable_and_precedes_disk_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.grim"
            base.write_bytes(b"archive")
            with (
                mock.patch.object(
                    feature_sum, "_feature_assembly_memory_limit_bytes",
                    return_value=1,
                ),
                mock.patch.object(
                    feature_sum, "_feature_assembly_disk_free_bytes"
                ) as disk_free,
            ):
                with self.assertRaisesRegex(
                    MemoryError, "peak RAM.*GHOST_MAX_SOLVE_GB"
                ):
                    feature_sum.preflight_feature_assembly_capacity(
                        str(base), str(root / "output.grim"),
                        radar_grid=SMALL_GRID,
                    )
            disk_free.assert_not_called()

    def test_disk_rejection_reports_atomic_scratch_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.grim"
            base.write_bytes(b"archive")
            with (
                mock.patch.object(
                    feature_sum, "_feature_assembly_memory_limit_bytes",
                    return_value=10 * 1024 ** 3,
                ),
                mock.patch.object(
                    feature_sum, "_feature_assembly_disk_free_bytes",
                    return_value=0,
                ),
            ):
                with self.assertRaisesRegex(OSError, "free scratch space") as raised:
                    feature_sum.preflight_feature_assembly_capacity(
                        str(base), str(root / "output.grim"),
                        radar_grid=SMALL_GRID,
                    )
            self.assertEqual(raised.exception.errno, errno.ENOSPC)

    def test_direct_build_capacity_failure_precedes_staging_and_keeps_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.grim"
            output = root / "assembled.grim"
            _write_empty_base(base)
            original = b"reviewed existing output"
            output.write_bytes(original)
            with (
                mock.patch.object(
                    feature_sum, "_feature_assembly_memory_limit_bytes",
                    return_value=1,
                ),
                mock.patch.object(feature_sum.tempfile, "mkstemp") as staging,
                mock.patch.object(feature_sum, "_load_grim") as full_load,
            ):
                with self.assertRaises(MemoryError):
                    feature_sum.add_features_to_monostatic_grim(
                        str(base), str(output), radar_grid=SMALL_GRID,
                        declared_coherent_base=True,
                    )
            staging.assert_not_called()
            full_load.assert_not_called()
            self.assertEqual(output.read_bytes(), original)

    def test_disk_is_rechecked_under_publication_lock_before_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.grim"
            output = root / "assembled.grim"
            _write_empty_base(base)
            original = b"reviewed existing output"
            output.write_bytes(original)
            with (
                mock.patch.object(
                    feature_sum, "_feature_assembly_memory_limit_bytes",
                    return_value=10 * 1024 ** 3,
                ),
                mock.patch.object(
                    feature_sum, "_feature_assembly_disk_free_bytes",
                    side_effect=[10 * 1024 ** 3, 0],
                ) as disk_free,
                mock.patch.object(feature_sum.tempfile, "mkstemp") as staging,
            ):
                with self.assertRaisesRegex(OSError, "free scratch space"):
                    feature_sum.add_features_to_monostatic_grim(
                        str(base), str(output), radar_grid=SMALL_GRID,
                        declared_coherent_base=True,
                    )
            self.assertEqual(disk_free.call_count, 2)
            staging.assert_not_called()
            self.assertEqual(output.read_bytes(), original)

    def test_radar_export_writes_coherent_role_without_archive_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "component.grim"
            with mock.patch.object(
                components, "tag_component",
                side_effect=AssertionError("redundant rewrite"),
            ) as tag_component:
                saved = feature_sum.export_radar_grim(
                    str(output), bor_result=None, placements=[], **SMALL_GRID
                )
            tag_component.assert_not_called()
            with np.load(saved, allow_pickle=False) as payload:
                self.assertEqual(str(payload["combine_role"]), "coherent")


if __name__ == "__main__":
    unittest.main()
