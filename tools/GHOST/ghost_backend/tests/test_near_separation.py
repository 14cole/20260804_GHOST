"""Independent close-gap quadrature regressions across assembly routes."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import hankel2

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ghost_backend.twod import solver as rcs
from ghost_backend.twod.assembly.separation import segment_distance


def pair_mesh(offset, gap, angle=0., length=1.):
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    panels=[]
    for x,y in ((0.,0.),(offset,gap)):
        a=rotation@np.array([x,y])*length
        b=rotation@np.array([x+1,y])*length
        panels.append(rcs.Panel('pair',2,0,0,0,a,b,(a+b)/2,
            rotation[:,0],rotation[:,1],length))
    return rcs._build_linear_mesh(panels)


def reference(offset,gap,k):
    def integrand(u):
        lo,hi=max(0.,offset-u),min(1.,offset+1-u)
        a,b=np.array([1.,0.]),np.array([-1.,1.])
        c,d=np.array([1+offset-u,u-offset]),np.array([-1.,1.])
        w=(np.outer(a,c)*(hi-lo)+(np.outer(a,d)+np.outer(b,c))*(hi**2-lo**2)/2
           +np.outer(b,d)*(hi**3-lo**3)/3)
        r=np.hypot(u,gap)
        return np.array([.25j*hankel2(0,k*r)*w,.25j*k*gap*hankel2(1,k*r)/r*w])
    return quad_vec(integrand,offset-1,offset+1,points=[offset,0,-gap,gap],
                    epsabs=1e-13,epsrel=1e-11)[0]


class SeparationTests(unittest.TestCase):
    def test_segment_geometry(self):
        self.assertAlmostEqual(float(segment_distance([0,0],[1,0],[.9,.01],[1.9,.01])),.01)
        self.assertEqual(float(segment_distance([0,0],[1,1],[0,1],[1,0])),0.)
        self.assertEqual(float(segment_distance([0,0],[1,0],[.5,0],[2,0])),0.)
        self.assertAlmostEqual(float(segment_distance([0,0],[0,0],[1,0],[2,0])),1.)

    def test_staggered_gap_matches_independent_integral(self):
        for gap in (.01,.001):
            ref_s,ref_k=reference(.9,gap,.3)
            for angle,length in ((0.,1.),(.47,.01)):
                mesh=pair_mesh(.9,gap,angle,length)
                s,k=rcs._assemble_linear_operator_matrices(mesh,.3/length,True)
                np.testing.assert_allclose(s[:2,2:],ref_s*length**2,rtol=2e-8,atol=1e-15)
                np.testing.assert_allclose(k[:2,2:],ref_k*length,rtol=2e-8,atol=1e-15)
                w=rcs._assemble_linear_hypersingular_matrix(mesh,.3/length)
                expected=-.3**2*ref_s+np.array([[1,-1],[-1,1]])*np.sum(ref_s)
                np.testing.assert_allclose(w[:2,2:],expected,rtol=2e-8,atol=1e-13)

    def test_almost_touching_collinear(self):
        mesh=pair_mesh(1.0001,0.)
        s,_=rcs._assemble_linear_operator_matrices(mesh,.3,True)
        np.testing.assert_allclose(s[:2,2:],reference(1.0001,0.,.3)[0],rtol=2e-8,atol=1e-13)


if __name__=='__main__': unittest.main()
