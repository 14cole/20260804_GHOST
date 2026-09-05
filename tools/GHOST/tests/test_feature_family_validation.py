import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"Backend"))
from feature_family_validation import study_template, validate_study, main
from test_feature_validation_cases import _write_field


class FamilyEvidenceTests(unittest.TestCase):
    def test_definition_cannot_be_mistaken_for_validated_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"study.json"
            path.write_text(json.dumps(study_template()))
            report=validate_study(path)
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["cases"]),13)
            self.assertTrue(all(row["status"]=="awaiting_reference_artifacts" for row in report["cases"]))

    def test_report_never_overwrites_an_existing_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'study.json'
            original=json.dumps(study_template())
            path.write_text(original)
            with self.assertRaises(FileExistsError):
                main(['--study',str(path),'--report',str(path)])
            self.assertEqual(path.read_text(),original)

    def test_three_level_complex_evidence_and_source_hashes_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); template=study_template()
            template["cases"]=template["cases"][:1]
            template["reference_solver"]="manufactured test fixture; not full-wave validation"
            case=template["cases"][0]
            (root/case["id"]).mkdir()
            clean=np.ones((3,2,1,3),complex)
            delta=np.full_like(clean,.2+.1j)
            all_paths=set(case["paths"].values())|{path for rows in case["reference_refinements"].values() for path in rows}
            for path in all_paths:
                _write_field(root/path,clean+delta if "featured" in path else clean,feature_response_sha256="a"*64 if "featured_prediction" in path else None)
            study=root/"study.json"; study.write_text(json.dumps(template))
            passed=validate_study(study)
            self.assertTrue(passed["passed"])
            self.assertEqual(len(passed["cases"][0]["artifact_sha256"]),8)
            _write_field(root/case["reference_refinements"]["featured"][1],-(clean+delta))
            failed=validate_study(study)
            self.assertFalse(failed["passed"])
            self.assertFalse(failed["cases"][0]["reference_convergence"]["featured"][1]["passed"])


if __name__=="__main__":
    unittest.main()
