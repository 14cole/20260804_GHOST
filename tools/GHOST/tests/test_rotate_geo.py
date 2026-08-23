#!/usr/bin/env python3
"""Regression checks for the hard-coded .geo rotation utility."""

import tempfile
import unittest
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Backend"))

from geometry_io import parse_geometry  # noqa: E402
from rotate_geo import rotate_geometry_file  # noqa: E402


SEMICIRCLE = """Title: simple profile
Segment: arc 2
properties: 2 0 0 0 0
-1 0 0 1
0 1 1 0
IBCS_Resistances:
7 constant 25 -10 0 0
Dielectrics:
3 2.5 -0.1 1.0 0.0
"""


FULL_CIRCLE = """Title: invalid full circle
Segment: circle 2
properties: 2 0 0 0 0
0 1 1 0
1 0 0 -1
0 -1 -1 0
-1 0 0 1
IBCS_Resistances:
Dielectrics:
"""


class RotateGeoTests(unittest.TestCase):
    def test_rotation_preserves_material_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.geo"
            output = Path(directory) / "rotated.geo"
            source.write_text(SEMICIRCLE, encoding="utf-8")
            _path, report = rotate_geometry_file(
                source,
                output,
                -90.0,
                validate_bor=True,
            )
            _title, segments, ibcs, dielectrics = parse_geometry(
                output.read_text(encoding="utf-8")
            )
            self.assertEqual(ibcs, [["7", "constant", "25", "-10", "0", "0"]])
            self.assertEqual(dielectrics, [["3", "2.5", "-0.1", "1.0", "0.0"]])
            self.assertGreaterEqual(report["min_x"], 0.0)
            self.assertAlmostEqual(segments[0].x[0], 0.0)
            self.assertAlmostEqual(segments[0].y[0], 1.0)

    def test_full_circle_is_rejected_for_bor(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "circle.geo"
            output = Path(directory) / "rotated.geo"
            source.write_text(FULL_CIRCLE, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "full circle"):
                rotate_geometry_file(
                    source,
                    output,
                    0.0,
                    validate_bor=True,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
