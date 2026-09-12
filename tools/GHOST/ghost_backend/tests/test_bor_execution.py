"""BOR batching, basis reuse, coefficient compression and resource contracts."""
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np
from scipy.sparse import csr_matrix
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.bor import solver as bor
from ghost_backend.bor.options import option_scope, validate_options, current_options
from ghost_backend.bor.tiled import primitive, modal_matrix


class BorExecutionTests(unittest.TestCase):
    def sweep(self, options, reduction=None):
        n = 96
        a = np.diag(np.linspace(1., 2., n)).astype(complex)
        base = np.column_stack((np.linspace(1., 2., n), np.cos(np.arange(n))))
        angles = np.linspace(0., 180., 97)
        calls = []
        def rhs(m, th, pol):
            return base @ np.array([np.cos(np.radians(th)), (1 if pol == 'VV' else 1j)*np.sin(np.radians(th))])
        def batch(m, th, pols):
            calls.append(len(th))
            return np.column_stack([rhs(m,t,p) for t in th for p in pols])
        def assemble(m):
            if reduction is None:
                return a, None
            return bor._reduce_constrained_operator(a, reduction), reduction
        with option_scope(validate_options(options)):
            f, _, stats = bor._mode_sweep(n,angles,('VV','HH'),0,1e-6,
                assemble,rhs,lambda m,x,th,p: x.sum(), rhs_batch=batch, monitor_cond=True)
        return f, stats, calls

    def test_batches_keep_one_factor_and_match_full_physical_sweep(self):
        reference, _, _ = self.sweep(dict(angle_batch_size=256,rhs_compression='off'))
        actual, stats, calls = self.sweep(dict(angle_batch_size=7,rhs_compression='off'))
        np.testing.assert_allclose(actual,reference,rtol=2e-13,atol=2e-13)
        event = stats['modal_execution']['systems'][0]
        self.assertEqual(event['factorizations'],1)
        self.assertEqual(event['rhs_batches'],14)
        self.assertLessEqual(max(calls),7)
        self.assertEqual(sum(calls),97)

    def test_basis_reuses_columns_across_batches_and_checks_physical_rhs(self):
        reference, _, _ = self.sweep(dict(rhs_compression='off'))
        actual, stats, _ = self.sweep(dict(angle_batch_size=32,rhs_compression='on'))
        np.testing.assert_allclose(actual,reference,rtol=2e-13,atol=2e-13)
        evidence=stats['modal_execution']['systems'][0]['sweep_compression']
        self.assertLess(evidence['solved_columns'],20)
        self.assertGreater(evidence['reused_basis_batches'],0)
        self.assertLessEqual(evidence['max_reconstructed_backward_error'],1e-12)

    def test_sparse_constraints_preserve_batched_fields(self):
        q=np.zeros((96,94),complex)
        q[:94]=np.eye(94)
        q[94,0]=1j
        q[95,1]=-1.
        reference, _, _ = self.sweep(dict(angle_batch_size=256,rhs_compression='off'),q)
        actual, _, _ = self.sweep(dict(angle_batch_size=9,rhs_compression='on'),csr_matrix(q))
        np.testing.assert_allclose(actual,reference,rtol=2e-13,atol=2e-13)

    def test_memory_tracks_batch_and_active_workers(self):
        with option_scope(validate_options(dict(angle_batch_size=8))):
            a=bor.estimate_bor_dense_peak_gb(1000,2000,workers=2,mode_tasks=4)
            b=bor.estimate_bor_dense_peak_gb(1000,16,workers=2,mode_tasks=4)
        self.assertEqual(a,b)
        with option_scope(validate_options(dict(angle_batch_size=64))):
            self.assertGreater(bor.estimate_bor_dense_peak_gb(1000,2000,workers=2),a)

    def test_mode_workers_keep_factorization_telemetry(self):
        from ghost_backend.execution.metrics import SolveMetrics, metrics_scope
        metrics=SolveMetrics()
        metrics.start()
        try:
            with metrics_scope(metrics):
                self.sweep(dict(angle_batch_size=7))
        finally:metrics.finish()
        self.assertEqual(metrics.calls['factorization'],1)
        self.assertGreater(metrics.calls['rhs_solve'],1)

    def test_compressed_inverse_beyond_leaf_size_and_transpose(self):
        from ghost_backend.bor.factor import compressed_factor
        from ghost_backend.bor.tiled import TileExpression
        n=320
        t=np.linspace(0,1,n)
        a=np.diag(1+t).astype(complex)+.01*np.exp(-abs(t[:,None]-t[None,:]))*(1+.2j)
        calls=[]
        def query(r,c):
            calls.append((len(r),len(c)))
            return a[np.ix_(r,c)].copy()
        oracle=TileExpression((n,n),query,t[:,None],t[:,None])
        factor=compressed_factor(oracle,1,True,validate_options(dict(factorization='compressed')),1)
        rhs=np.column_stack((np.sin(t),np.cos(t)*(1+1j)))
        for trans,matrix in ((0,a),(1,a.T),(2,a.conj().T)):
            value=factor.inverse(rhs,trans=trans)
            np.testing.assert_allclose(matrix@value,rhs,rtol=1e-11,atol=1e-12)
        self.assertLessEqual(max(max(pair) for pair in calls),32)
        self.assertLess(factor.condition,1e12)
        self.assertGreater(factor.a.compressed,0)

    def test_compressed_budget_and_cancellation_fail_without_dense_fallback(self):
        from ghost_backend.bor.factor import compressed_factor
        from ghost_backend.bor.tiled import TileExpression
        oracle=TileExpression((256,256),lambda r,c:(r[:,None]==c).astype(complex))
        options=validate_options(dict(factorization='compressed',compressed_storage_mib=16))
        with self.assertRaises(MemoryError):
            compressed_factor(oracle,0,False,options,100000)
        def cancel():raise RuntimeError('cancel coefficient assembly')
        with self.assertRaisesRegex(RuntimeError,'cancel coefficient assembly'):
            compressed_factor(oracle,0,False,options,1,cancel)

    def test_invalid_options_and_precision_reject_before_assembly(self):
        for invalid in ({'angle_batch_size':True},{'factorization':'auto'},{'compression_tile':0},{'unknown':1}):
            with self.assertRaises(ValueError):validate_options(invalid)
        with self.assertRaisesRegex(ValueError,'double precision'):
            bor.solve_bor(None,1e9,[0],table_precision='single',bor_options={'factorization':'compressed'})
        self.assertEqual(current_options()['factorization'],'dense')

    def test_cancel_between_batches(self):
        cancelled=[False]
        def checkpoint():
            if cancelled[0]:raise RuntimeError('cancel requested')
        def farfield(*args):
            cancelled[0]=True
            return 1.
        with option_scope(validate_options(dict(angle_batch_size=1))):
            with self.assertRaisesRegex(RuntimeError,'cancel requested'):
                bor._mode_sweep(1,[0.,20.],['VV'],0,1e-6,lambda m:(np.ones((1,1),complex),None),
                    lambda *args:np.ones(1,complex),farfield,check_abort=checkpoint)

    def test_cross_tiles_match_original_complex_medium_operators(self):
        sp=bor.BorPecSolver(bor.sphere_generatrix(.035,10),1e9,gauss_order=3,medium=(2.5-.05j,1.))
        sq=bor.BorPecSolver(bor.sphere_generatrix(.02,8),1e9,gauss_order=3,medium=(2.5-.05j,1.))
        cross=bor.BorCrossOperators(sp,sq)
        cross.prepare(2)
        rows=np.arange(2*sp.Nn)[::2]
        cols=np.arange(2*sq.Nn)[1::2]
        for m in (0,1,-1,2):
            for kind,method in [('T',cross.assemble_T),('P',cross.assemble_P)]:
                expected=method(m,2)[np.ix_(rows,cols)]
                actual=primitive(cross,kind,m,2).get(rows,cols)
                np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-12*max(np.max(abs(expected)),1e-30))

    def test_compressed_physics_avoids_full_far_matrices(self):
        points=bor.sphere_generatrix(.025,12)
        cases=[(bor.solve_bor,dict(formulation='cfie')),
               (bor.solve_bor,dict(formulation='cfie',zs=100j)),
               (bor.solve_bor,dict(sheet_zs=100+20j)),
               (bor.solve_bor_dielectric,dict(eps_r=2.5-.05j))]
        for fn,kwargs in cases:
            reference=fn(points,1e9,[0.,40.,90.,180.],gauss_order=3,workers=2,**kwargs)
            with mock.patch.object(bor.BorPecSolver,'_ensure_dense_point_matrices',side_effect=AssertionError('dense far matrix')):
                actual=fn(points,1e9,[0.,40.,90.,180.],gauss_order=3,workers=2,
                          bor_options=dict(factorization='compressed',angle_batch_size=2),**kwargs)
            for pol in ('amp_vv','amp_hh'):
                np.testing.assert_allclose(actual[pol],reference[pol],rtol=2e-9,atol=2e-11)
            self.assertEqual(actual['assembly'],'compressed')
            self.assertTrue(actual['mode_converged'])
            self.assertLessEqual(actual['linear_backward_error'],1e-12)


if __name__=='__main__':unittest.main()
