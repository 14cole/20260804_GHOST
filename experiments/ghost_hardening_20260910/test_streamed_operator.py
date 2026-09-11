from streamed_operator import *
import unittest
from unittest.mock import patch
from scipy.special import hankel2
from checked_compressed_system import CompressedSystem,Oracle
import gc,weakref


class StreamedTests(unittest.TestCase):
    def test_checked_inverse_caps_cancellation_transposes_and_input_release(self):
        rng=np.random.RandomState(551)
        n=260
        a=np.eye(n)*20+rng.randn(n,3)@(rng.randn(3,n)+1j*rng.randn(3,n))/n
        coordinates=np.arange(n)[:,None].astype(float)
        class ExactOracle(Oracle):
            dropped_routes=0
            def get_with_error(self,rows,cols):
                value=self.get(rows,cols)
                return value,np.zeros(value.shape)
        oracle=ExactOracle(a)
        operator=StreamedOperator(oracle,coordinates,tile=64)
        refs=[weakref.ref(oracle),weakref.ref(operator),weakref.ref(coordinates)]
        canceled=[False]
        def check():
            if canceled[0]:raise InterruptedError('canceled')
        system=CompressedSystem(operator,coordinates,checkpoint=check,tolerance=1e-13)
        self.assertLess(system.bytes,2*a.nbytes)
        row_error=operator.row_error
        column_error=np.zeros(n)
        for j in range(0,n,32):
            cols=np.arange(j,min(n,j+32))
            column_error[cols]=np.sum(abs(a[:,cols]-operator.get(np.arange(n),cols)),axis=0)
        with self.assertRaises(MemoryError):
            CompressedSystem(operator,coordinates,budget=1)
        for invalid in (np.nan,np.inf,-1):
            with self.assertRaises(ValueError):CompressedSystem(operator,coordinates,budget=invalid)
            with self.assertRaises(ValueError):StreamedOperator(oracle,coordinates,budget=invalid)
        for invalid in (0,-1,1025,1.5):
            with self.assertRaises(ValueError):StreamedOperator(oracle,coordinates,tile=invalid)
        for rows in ([-1],[n],[0,0],[.1]):
            with self.assertRaises(ValueError):operator.get(rows,[0])
        enabled=gc.isenabled();gc.disable()
        try:
            del operator,oracle,coordinates
            self.assertTrue(all(ref() is None for ref in refs))
        finally:
            if enabled:gc.enable()
        b=rng.randn(n,4)+1j*rng.randn(n,4)
        initial=np.zeros_like(b)
        system.solve_checked(b,input_error=row_error,initial=initial)
        np.testing.assert_array_equal(initial,0)
        adopted=system.solve_checked(b,input_error=row_error,initial=initial,overwrite_initial=True)
        self.assertIs(adopted,initial)
        np.testing.assert_allclose(adopted,np.linalg.solve(a,b),rtol=2e-12,atol=2e-14)
        for trans,mat in ((0,a),(1,a.T),(2,a.conj().T)):
            x=system.solve_checked(b,trans=trans,input_error=row_error if trans==0 else column_error)
            np.testing.assert_allclose(x,np.linalg.solve(mat,b),rtol=2e-12,atol=2e-14)
            self.assertLessEqual(system.evidence['last_original_residual_bound'],1e-12)
        with self.assertRaises(RuntimeError):system.solve_checked(b,input_error=np.ones(n)*10)
        for invalid in (np.ones(n-1),np.ones((n,0)),np.full(n,np.nan)):
            with self.assertRaises(ValueError):system.apply(invalid,solve=True)
            with self.assertRaises(ValueError):system.solve_checked(invalid)
        with self.assertRaises(ValueError):system.apply(b,trans=3)
        canceled[0]=True
        with self.assertRaises(InterruptedError):system.apply(b,solve=True)
        with self.assertRaises(InterruptedError):system.apply(b)
        with self.assertRaises(InterruptedError):CompressedSystem(ExactOracle(a),np.arange(n)[:,None],checkpoint=check)
        with self.assertRaises(la.LinAlgWarning):CompressedSystem(ExactOracle(np.zeros((16,16))),np.arange(16)[:,None])

    def test_compression_preserves_high_rank_tiles_and_bounds_roundoff(self):
        rng=np.random.RandomState(77)
        for raw in (rng.randn(40,40)+1j*rng.randn(40,40),
                    rng.randn(40,3)@rng.randn(3,40),np.zeros((40,40))):
            raw=raw.astype(complex)
            for method in ('svd','qr'):
                payload,approx,error,compressed=tile_payload(raw,np.zeros(raw.shape),1e-14,method)
                np.testing.assert_array_less(abs(raw-approx)-error,1e-30)
                if compressed:self.assertLess(sum(p.nbytes for p in payload),raw.nbytes)

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
        for kind in ('layered','coated','mixed','pec_ibc','equal_k'):
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
