"""Convention evidence stays distinct from numerical eligibility."""

import unittest

from grim_metadata import inspect_scalar_metadata


class ScalarMetadataTests(unittest.TestCase):
    def test_conflicting_advisory_declarations_remain_inspectable(self):
        evidence = inspect_scalar_metadata(
            "phase_reference", {"phase_reference": "nose"},
            {"phase_reference": "scene center"},
        )
        self.assertEqual(evidence.status, "conflicting")
        self.assertEqual(evidence.sources, ("units", "extra"))
        self.assertEqual(evidence.declarations, ("nose", "scene center"))
        self.assertEqual(evidence.scalar(advisory=True), "")
        with self.assertRaisesRegex(ValueError, "contradictory phase_reference"):
            evidence.scalar(advisory=False)

    def test_malformed_and_missing_have_different_evidence(self):
        malformed = inspect_scalar_metadata(
            "time_convention", {"time_convention": ["a", "b"]},
            {"time_convention": "exp(+jwt)"},
        )
        self.assertEqual(malformed.status, "malformed")
        self.assertEqual(malformed.malformed_sources, ("units",))
        self.assertEqual(malformed.scalar(advisory=True), "exp(+jwt)")
        with self.assertRaisesRegex(ValueError, "must be scalar"):
            malformed.scalar(advisory=False)
        self.assertEqual(inspect_scalar_metadata("time_convention", {}, {}).status, "missing")

    def test_equivalent_time_conventions_are_consistent(self):
        evidence = inspect_scalar_metadata(
            "time_convention", {"time_convention": "exp(+j ω t)"},
            {"time_convention": "exp(jwt)"},
        )
        self.assertEqual(evidence.status, "consistent")
        self.assertEqual(evidence.scalar(advisory=False), "exp(+j ω t)")

    def test_false_and_zero_are_not_missing(self):
        for value in (False, 0):
            with self.subTest(value=value):
                evidence = inspect_scalar_metadata("motion_compensated", {}, {"motion_compensated": value})
                self.assertEqual(evidence.status, "consistent")
                self.assertEqual(evidence.scalar(advisory=False), str(value))


if __name__ == "__main__":
    unittest.main()
