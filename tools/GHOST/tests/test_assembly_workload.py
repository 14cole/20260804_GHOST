"""Operation-count and review-gate regressions for full-vehicle Assembly."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1] / "Backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from assembly_workload import (  # noqa: E402
    ASSEMBLY_REVIEW_POINT_FIELD_CELLS,
    ASSEMBLY_REVIEW_SHADOW_RAYS,
    estimate_assembly_workload,
    warnings_require_workload_acknowledgement,
    workload_review_warning,
)


class AssemblyWorkloadTests(unittest.TestCase):
    def test_counts_match_cached_shadow_and_frequency_dependent_field_work(self):
        result = estimate_assembly_workload(
            look_count=361,
            frequency_count=19,
            point_count=800,
            line_path_count=12,
            line_segment_count=90,
            line_piece_count=20_000,
            mesh_triangle_count=800_000,
            shadow_enabled=True,
            quantities_validated=True,
            line_piece_count_exact=True,
            mesh_triangle_count_exact=True,
        )

        self.assertEqual(result.radar_grid_cell_count, 361 * 19)
        self.assertEqual(result.point_field_cell_count, 361 * 19 * 800)
        self.assertEqual(result.line_field_cell_count, 361 * 19 * 20_000)
        # Visibility is frequency-independent and cached once.
        self.assertEqual(result.shadow_ray_upper_bound, 361 * (800 + 20_000))
        self.assertEqual(
            result.packed_visibility_bytes_upper_bound,
            (800 + 20_000) * ((361 + 7) // 8),
        )

    def test_review_threshold_is_inclusive_and_warning_makes_no_eta_claim(self):
        result = estimate_assembly_workload(
            look_count=1,
            frequency_count=1,
            point_count=ASSEMBLY_REVIEW_POINT_FIELD_CELLS,
            line_path_count=0,
            line_segment_count=0,
            line_piece_count=0,
            quantities_validated=True,
        )
        warning = workload_review_warning(result)

        self.assertIsNotNone(warning)
        self.assertIn("operation counts only", warning)
        self.assertIn("no elapsed-time estimate", warning)
        self.assertNotIn("hour", warning.casefold())
        self.assertTrue(warnings_require_workload_acknowledgement((warning,)))

    def test_prevalidation_never_creates_a_sealed_warning(self):
        result = estimate_assembly_workload(
            look_count=ASSEMBLY_REVIEW_SHADOW_RAYS,
            frequency_count=1,
            point_count=1,
            line_path_count=0,
            line_segment_count=0,
            line_piece_count=0,
            shadow_enabled=True,
            quantities_validated=False,
        )
        self.assertTrue(result.review_reasons)
        self.assertIsNone(workload_review_warning(result))


if __name__ == "__main__":
    unittest.main()
