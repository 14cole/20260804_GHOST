#!/usr/bin/env python3
"""Release regressions for shared solve-quality gates."""

import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

from solver_quality import evaluate_mesh_convergence  # noqa: E402


def _sample(angle, amplitude):
    power = float(abs(amplitude) ** 2)
    return {
        "frequency_ghz": 2.0,
        "theta_inc_deg": float(angle),
        "theta_scat_deg": float(angle),
        "rcs_linear": power,
        "rcs_amp_real": float(complex(amplitude).real),
        "rcs_amp_imag": float(complex(amplitude).imag),
    }


class MeshConvergenceMatchingTests(unittest.TestCase):
    def _evaluate(self, base_samples, fine_samples):
        return evaluate_mesh_convergence(
            {"samples": base_samples},
            {"samples": fine_samples},
            rms_limit_db=1.0e-12,
            max_abs_limit_db=1.0e-12,
            complex_rms_limit=1.0e-12,
            complex_max_limit=1.0e-12,
            phase_rms_limit_deg=1.0e-12,
            phase_max_limit_deg=1.0e-12,
        )

    def test_distinct_angles_colliding_at_key_precision_are_all_compared(self):
        angles = (12.3639797386, 12.3639797394)
        base = [_sample(angles[0], 1.0 + 0.25j),
                _sample(angles[1], 2.0 - 0.5j)]
        # Deliberately reverse serialization order.  Matching must use the
        # exact coordinates within the shared rounded-key bucket.
        fine = [_sample(angles[1], 2.0 - 0.5j),
                _sample(angles[0], 1.0 + 0.25j)]

        result = self._evaluate(base, fine)

        self.assertTrue(result["passed"])
        self.assertEqual(result["sample_count"], 2)

    def test_duplicate_exact_coordinates_remain_invalid(self):
        duplicate = _sample(12.0, 1.0 + 0.0j)
        with self.assertRaisesRegex(ValueError, "duplicate exact sample"):
            self._evaluate([duplicate, dict(duplicate)], [duplicate])

    def test_rounded_bucket_multiplicity_must_match(self):
        angles = (12.3639797386, 12.3639797394)
        base = [_sample(angles[0], 1.0), _sample(angles[1], 2.0)]
        fine = [_sample(angles[0], 1.0)]
        with self.assertRaisesRegex(
            ValueError, "1 missing and 0 extra fine-result"
        ):
            self._evaluate(base, fine)


if __name__ == "__main__":
    unittest.main()
