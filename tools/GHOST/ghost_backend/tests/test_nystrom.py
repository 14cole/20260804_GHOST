"""Independent cylinder reference and noncircular spectral convergence."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ghost_backend.twod.nystrom import ellipse,solve_smooth_pec
from ghost_backend.validation.cylinder import pec_cylinder_backscatter_amplitude


class NystromTests(unittest.TestCase):
    def test_circle_both_polarizations_and_dirichlet_resonance(self):
        for k in (1.,5.,10.,30.):
            for pol in ('TM','TE'):
                value=solve_smooth_pec(ellipse(.3,.3),256,k/.3,[0,23,91],pol)
                expected=pec_cylinder_backscatter_amplitude(.3,k/.3*299792458/(2*np.pi),pol)
                np.testing.assert_allclose(value['amplitude'],expected,rtol=1e-10,atol=1e-11)
        k=2.404825557695773
        value=solve_smooth_pec(ellipse(1.,1.),64,k,[0],'TM')
        expected=pec_cylinder_backscatter_amplitude(1.,k*299792458/(2*np.pi),'TM')
        np.testing.assert_allclose(value['amplitude'],expected,rtol=1e-10,atol=1e-11)

    def test_ellipse_refinement_and_translation_phase(self):
        angles=np.array([0,19,80,170]);k=8.
        shift=np.array([.3,-.5])
        a=solve_smooth_pec(ellipse(1.,.6),128,k,angles)
        b=solve_smooth_pec(ellipse(1.,.6),256,k,angles)
        c=solve_smooth_pec(ellipse(1.,.6,shift),256,k,angles)
        np.testing.assert_allclose(a['amplitude'],b['amplitude'],rtol=1e-9,atol=1e-9)
        direction=np.column_stack((np.cos(np.deg2rad(angles)),np.sin(np.deg2rad(angles))))
        np.testing.assert_allclose(c['amplitude'],b['amplitude']*np.exp(2j*k*(direction@shift)),rtol=1e-10)

    def test_reject_invalid_geometry(self):
        with self.assertRaises(ValueError):solve_smooth_pec(ellipse(1,1),17,3,[0])
        with self.assertRaises(ValueError):solve_smooth_pec(ellipse(1,1),32,3,[float('nan')])

if __name__=='__main__':unittest.main()
