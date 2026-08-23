from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ibc.io import MATERIAL_HEADER, read_material_table
from tools.convert_nist_bam_pdms import SOURCE_SHA256, convert, sha256, verify_sources
from tools.validate_material_mix import (
    FIT_RMS_LIMIT_PERCENT,
    INVERSE_FRACTION_TOLERANCE,
    MEASURED_MEDIAN_LIMIT_PERCENT,
    MEASURED_P95_LIMIT_PERCENT,
    validate,
)


FREDDY_ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = FREDDY_ROOT / "materials" / "validation" / "nist_bam_pdms"
SOURCE_DIR = PACK_ROOT / "source"
CONVERTED_DIR = PACK_ROOT / "freddy"


class NistMaterialValidationTests(unittest.TestCase):
    def test_vendored_sources_match_nist_checksums(self) -> None:
        verify_sources(SOURCE_DIR)
        for filename, expected in SOURCE_SHA256.items():
            self.assertEqual(sha256(SOURCE_DIR / filename), expected)

    def test_conversion_is_reproducible_and_solver_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            generated = Path(tmp) / "freddy"
            manifest = convert(SOURCE_DIR, generated)
            expected_names = set(manifest["outputs"])
            self.assertEqual(
                expected_names,
                {path.name for path in CONVERTED_DIR.glob("*.csv")},
            )
            for filename in expected_names:
                with self.subTest(filename=filename):
                    generated_bytes = (generated / filename).read_bytes()
                    checked_in_bytes = (CONVERTED_DIR / filename).read_bytes()
                    self.assertEqual(generated_bytes, checked_in_bytes)
                    self.assertEqual(
                        generated_bytes.splitlines()[0].decode("ascii"),
                        MATERIAL_HEADER,
                    )
                    table = read_material_table(generated / filename)
                    self.assertTrue(all(value.imag <= 0.0 for value in table.eps_r))
                    self.assertTrue(all(value.imag <= 0.0 for value in table.mu_r))

    def test_forward_inverse_and_measured_regressions(self) -> None:
        report = validate(CONVERTED_DIR)
        self.assertTrue(report["passed"], report["failures"])
        for result in report["forward_fit_validation"].values():
            self.assertLess(result["rms_percent"], FIT_RMS_LIMIT_PERCENT)
        for result in report["inverse_fit_validation"].values():
            self.assertLess(
                result["absolute_fraction_error"], INVERSE_FRACTION_TOLERANCE
            )
        for result in report["measured_validation"].values():
            self.assertLess(
                result["median_percent"], MEASURED_MEDIAN_LIMIT_PERCENT
            )
            self.assertLess(result["p95_percent"], MEASURED_P95_LIMIT_PERCENT)


if __name__ == "__main__":
    unittest.main()
