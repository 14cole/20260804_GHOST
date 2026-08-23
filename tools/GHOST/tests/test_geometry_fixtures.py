#!/usr/bin/env python3
"""Integrity checks for geometry inputs used by equivalence benchmarks."""

import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Backend"))

from geometry_io import parse_geometry  # noqa: E402


FIXTURES = (
    "coupon_frd.geo",
    "coupon_opn_010.geo",
    "coupon_opn_017.geo",
)


class GeometryFixtureTests(unittest.TestCase):
    def test_equivalence_coupon_fixtures_exist_and_parse(self):
        root = REPO / "tests" / "fixtures" / "geometries"
        for name in FIXTURES:
            with self.subTest(name=name):
                path = root / name
                self.assertTrue(path.is_file(), f"missing fixture: {path}")
                title, segments, _ibcs, _dielectrics = parse_geometry(
                    path.read_text(encoding="utf-8")
                )
                self.assertEqual(title, "coupon")
                self.assertEqual(len(segments), 1)
                self.assertGreater(len(segments[0].x), 1)
                self.assertEqual(len(segments[0].x), len(segments[0].y))


if __name__ == "__main__":
    unittest.main()
