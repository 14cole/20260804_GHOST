from streamed_operator import *
import unittest
from unittest.mock import patch
from scipy.special import hankel2


class StreamedTests(unittest.TestCase):
    def test_kernel_envelope_covers_oscillatory_and_damped_values(self):
        for k in (1-100j,100-1j,10-10j,30-4000j):
            r=np.geomspace(2.01/(-k.imag),500/(-k.imag),300)
            g,h=hankel_envelopes(k,r)
            self.assertTrue(np.all(abs(.25j*hankel2(0,k*r))<=g))
            self.assertTrue(np.all(abs(.25j*k*hankel2(1,k*r))<=h))
        self.assertIsNone(hankel_envelopes(10.,np.array([100.])))
        self.assertIsNone(hankel_envelopes(10+1j,np.array([100.])))
        self.assertIsNone(hankel_envelopes(10-1j,np.array([.1])))

    def test_geometry_tiles_and_transposes_match_independent_matrix(self):
        rng=np.random.RandomState(971)
        for kind in ('layered','coated','mixed'):
            for pol in ('TE','TM'):
                mesh,infos,k=prepare(kind,pol,count=24)
                reference,layout=mr._assemble_system_fresh(mesh,infos,pol,8,8)
                oracle=PreparedOracle(mesh,infos,pol,cut=32)
                xy=np.empty((oracle.n,2))
                for (mi,side),(start,count) in layout['dof_map'].items():
                    xy[start:start+count]=oracle.xy[layout['ifaces'][mi]['nodes']]
                original_zeros=np.zeros
                def bounded(shape,*args,**kwargs):
                    if isinstance(shape,tuple) and shape==(oracle.n,oracle.n):
                        raise AssertionError('Full system allocation during streamed assembly')
                    return original_zeros(shape,*args,**kwargs)
                with patch.object(np,'zeros',bounded):
                    operator=StreamedOperator(oracle,xy,tile=32)
                b=rng.randn(oracle.n,2)+1j*rng.randn(oracle.n,2)
                for trans,a in ((0,reference),(1,reference.T),(2,reference.conj().T)):
                    np.testing.assert_allclose(operator.matmul(b,trans),a@b,rtol=3e-12,atol=2e-15)
                selected=rng.choice(oracle.n,16,False)
                got=operator.get(selected,selected[::-1])
                np.testing.assert_allclose(got,reference[np.ix_(selected,selected[::-1])],rtol=3e-12,atol=2e-15)
                self.assertEqual(oracle.entries,oracle.n**2)
                with self.assertRaises(MemoryError):
                    StreamedOperator(oracle,xy,tile=32,budget=1)


if __name__=='__main__':unittest.main()
