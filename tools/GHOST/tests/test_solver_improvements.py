"""Numerical guarantees for opt-in refinement, telemetry, and interference."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"Backend"))
import rcs_solver as rcs
from refined_lu import RefinedLU, linear_precision, requested_precision
from assembly_inspector import interference_metrics, _stored_complex_sample
from mesh_guidance import refined_density, geometry_refinement_candidates
from solver_quality import accuracy_target_policy
from test_thin_sheet import sheet_snapshot


class MixedPrecisionTests(unittest.TestCase):
    def test_multiple_rhs_and_adjoint_reach_double_residual_accuracy(self):
        rng = np.random.default_rng(761)
        a = rng.normal(size=(96,96))+1j*rng.normal(size=(96,96))+20*np.eye(96)
        b = rng.normal(size=(96,4))+1j*rng.normal(size=(96,4))
        factor = RefinedLU(a)
        for trans, op in ((0,a),(1,a.T),(2,a.conj().T)):
            actual = factor.solve(b, trans=trans)
            np.testing.assert_allclose(actual, np.linalg.solve(op,b), rtol=1e-10, atol=1e-11)
        self.assertEqual(factor.lu.dtype, np.complex64)
        self.assertGreaterEqual(factor.max_corrections, 1)

    def test_single_precision_singularity_falls_back_without_changing_answer(self):
        a = np.array([[1,1],[1,1+1e-10]], complex)
        b = np.array([2,2+1e-10], complex)
        rcs._reset_dense_backend_telemetry()
        with linear_precision("mixed"):
            actual = rcs._solve_dense_system(a,b)
        np.testing.assert_allclose(actual,np.linalg.solve(a,b),rtol=1e-10)
        self.assertTrue(any("fell back" in reason for reason in rcs._dense_backend_summary()["dense_fallback_reasons"]))
        self.assertEqual(requested_precision(),"double")

    def test_mixed_and_double_thin_layer_phase_and_condition_agree(self):
        snapshot = sheet_snapshot([[-.05,0.],[.05,0.]],24)
        snapshot.update(ibcs=[["1","thin_dielectric",".0005","2"]],dielectrics=[["2","3","-.05","1","0"]])
        kwargs = dict(geometry_units="meters", compute_condition_number=True)
        double = rcs.solve_monostatic_rcs_2d(snapshot,[1.],[12.,48.,86.],**kwargs)
        with linear_precision("mixed"):
            mixed = rcs.solve_monostatic_rcs_2d(snapshot,[1.],[12.,48.,86.],**kwargs)
        def field(result):
            return [complex(r["rcs_amp_real"],r["rcs_amp_imag"]) for r in result["samples"]]
        np.testing.assert_allclose(field(mixed),field(double),rtol=1e-9,atol=1e-12)
        self.assertEqual(mixed["metadata"]["linear_backend"],"cpu_mixed_lu")
        self.assertIn("mixed_factorization",mixed["metadata"]["runtime_profile"]["stage_seconds"])
        self.assertTrue(mixed["metadata"]["condition_est_computed"])


class InterferenceTests(unittest.TestCase):
    def test_cancellation_is_not_ranked_by_feature_power(self):
        metrics=interference_metrics(1.,[-.9, .05j])
        self.assertLess(metrics["removal_change_m2"][0],0)
        self.assertLess(metrics["interference_m2"][0],0)
        expected=4*np.pi*(abs(.1+.05j)**2-abs(1+.05j)**2)
        self.assertAlmostEqual(metrics["removal_change_m2"][0],expected)

    def test_gain_and_phase_sensitivities_match_finite_differences(self):
        fields=np.array([.1+.2j,-.7-.1j]); body=.8+.6j; h=1e-5
        base=interference_metrics(body,fields)
        for i in range(2):
            for key, phase in (("gain_derivative_m2",False),("phase_derivative_m2_per_deg",True)):
                gp,gm=np.ones(2,complex),np.ones(2,complex)
                gp[i]=np.exp(1j*np.radians(h)) if phase else 1+h
                gm[i]=np.exp(-1j*np.radians(h)) if phase else 1-h
                derivative=(interference_metrics(body,fields,gp)["sigma_total"]-interference_metrics(body,fields,gm)["sigma_total"])/(2*h)
                self.assertAlmostEqual(derivative,base[key][i],places=7)

    def test_complex_archive_sample_keeps_phase_and_exact_axes(self):
        from test_feature_validation_cases import _write_field
        data=(np.arange(18).reshape(3,2,1,3)+1)*(1+.4j)
        with tempfile.TemporaryDirectory() as temp:
            path=_write_field(Path(temp)/"field.grim",data)
            np.testing.assert_array_equal(_stored_complex_sample(path,9.5,120.,20.),data[1,1,0])
            with self.assertRaises(ValueError):
                _stored_complex_sample(path,9.5,125.,20.)


class MeshGuidanceTests(unittest.TestCase):
    def test_selected_density_preserves_mode_and_only_targets_local_geometry(self):
        self.assertEqual([refined_density(n) for n in (0,3,-25)], ["-40","6","-50"])
        with self.assertRaises(ValueError):
            refined_density("2.5")
        snapshot=sheet_snapshot([[0.,0.],[1.,0.],[1.,1.]])
        candidates=geometry_refinement_candidates(snapshot["segments"])
        self.assertEqual(candidates,{0:["corner","open end"]})
        self.assertLess(accuracy_target_policy("tight")["complex_max_limit"],accuracy_target_policy()["complex_max_limit"])


if __name__ == "__main__":
    unittest.main()
