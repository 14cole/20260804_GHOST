"""Accuracy, ownership and bounded sweep reuse for the matrix pipeline."""
import os
import sys
import weakref
import gc
from pathlib import Path
import unittest
from unittest import mock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Backend'))
import ghost_backend.twod.solver as rcs
import ghost_backend.twod.operators as ops
import ghost_backend.twod.assembly.kernels as kernels
import ghost_backend.execution.cpu as execution
from ghost_backend.linalg.dense import DenseFactor
from ghost_backend.linalg.workspace import matrix_inf_norm
from ghost_backend.linalg.refined_lu import RefinedLU, linear_precision
from ghost_backend.linalg.hierarchical import HierarchicalFactor
from ghost_backend.linalg.sweep import SweepBasis, solve
from test_compact_multi_region import prepared


class SolverEfficiencyTests(unittest.TestCase):
    def test_complex_frobenius_norm_handles_layouts(self):
        from ghost_backend.linalg.sweep import _frobenius_norm
        rng=np.random.RandomState(237)
        a=rng.randn(85,39)+1j*rng.randn(85,39)
        for value in (a,np.asfortranarray(a),a.T,a[::2,::3],a[:0]):
            self.assertAlmostEqual(_frobenius_norm(value),np.linalg.norm(value),places=12)

    def test_prepared_geometry_matches_queries_and_rejects_other_mesh(self):
        from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
        for kind in ('pec', 'ibc', 'layered', 'mixed'):
            mesh, _, k = prepared(kind, 'TE', 20)
            geometry = AssemblyGeometry(mesh)
            rows = np.arange(len(mesh.nodes))[::2]
            cols = np.arange(len(mesh.nodes))[1::2]
            kwargs = dict(output_node_ids_many=[(rows, cols)])
            expected = ops._assemble_linear_operator_matrices_multi(mesh, k, True, [None], **kwargs)[0]
            actual = ops._assemble_linear_operator_matrices_multi(mesh, k, True, [None],
                prepared_geometry=geometry, **kwargs)[0]
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a.values, b.values)
            other, _, _ = prepared(kind, 'TE', 20)
            with self.assertRaisesRegex(ValueError, 'another mesh'):
                ops._assemble_linear_operator_matrices_multi(other, k, True, [None], prepared_geometry=geometry)

    def test_sweep_state_cannot_reuse_another_factor_or_retain_it(self):
        rng = np.random.RandomState(207)
        b = rng.randn(64, 2) @ rng.randn(2, 40)
        with mock.patch.dict(os.environ, {'GHOST_CPU_RHS_COMPRESSION':'on', 'GHOST_CPU_FACTORIZATION':'dense'}):
            state = SweepBasis(40)
            first = DenseFactor(np.eye(64, dtype=complex))
            solve(first, b, state)
            ref = weakref.ref(first)
            del first
            self.assertIsNone(ref())
            second = DenseFactor(np.eye(64, dtype=complex)*2)
            np.testing.assert_allclose(solve(second, b, state), b/2, atol=2e-13)
            self.assertEqual(second.event['sweep_compression']['fallback_columns'], 0)
            with mock.patch('ghost_backend.linalg.sweep._qr_basis', side_effect=AssertionError('unneeded QR')):
                np.testing.assert_allclose(solve(second, b, state), b/2, atol=2e-13)
            for invalid in (np.ones(64), np.empty((64, 0)), np.ones((63, 40)),
                            np.full((64, 40), np.nan), np.full((64, 40), np.inf)):
                with self.assertRaises(ValueError):
                    solve(second, invalid, state)
            # New incident columns can be much larger than the previous row scale.
            state.scale.fill(1e-300)
            np.testing.assert_allclose(solve(second, b*1e10, state), b*5e9, atol=1e-4)
            self.assertIsNone(state.scale)

    def test_hierarchical_geometry_and_rejected_workspace_release_before_lu(self):
        import ghost_backend.linalg.hierarchical as hf
        enabled = gc.isenabled()
        gc.disable()
        try:
            a = np.eye(260, dtype=complex)
            coordinates = np.arange(260)[:, None].astype(float)
            ref = weakref.ref(coordinates)
            factor = hf.HierarchicalFactor(a, coordinates)
            del coordinates
            self.assertIsNone(ref())
            refs = []
            def reject(*args, **kwargs):
                work = np.ones((128, 128), complex)
                refs.append(weakref.ref(work))
                raise hf.HierarchicalRejected('rank cap')
            original = hf.la.lu_factor
            def checked(*args, **kwargs):
                self.assertIsNone(refs[0]())
                return original(*args, **kwargs)
            with mock.patch.object(hf, 'compress', reject), mock.patch.object(hf.la, 'lu_factor', checked):
                factor = hf.HierarchicalFactor(a)
            np.testing.assert_array_equal(factor.solve(np.ones(260)), 1)
        finally:
            if enabled:
                gc.enable()

    def test_rejected_kernel_workspace_released_before_degree_retry(self):
        mesh, _, _ = prepared('pec', 'TE', 20)
        refs = []
        original = kernels.KernelTable
        def candidate(k, upper, degree=12):
            if degree == 12:
                work = np.ones(2048, complex)
                refs.append(weakref.ref(work))
                raise kernels.Rejected('degree test')
            self.assertIsNone(refs[0]())
            return original(k, upper, degree=degree)
        with execution._STATE.override(execution.CPUState()), mock.patch.object(kernels, 'KernelTable', candidate):
            fg, _ = kernels.select_far_kernels(mesh, 10-1j, ops._far_green_into, ops._far_hankel1_into)
        self.assertTrue(hasattr(fg, 'pair'))

    def test_positional_destination_is_never_frozen_by_operator_cache(self):
        mesh,_,k=prepared('pec','TE',20)
        destination=np.zeros((len(mesh.nodes),len(mesh.nodes)),complex,order='F')
        state=execution.CPUState()
        with execution._STATE.override(state):
            actual=ops._assemble_linear_hypersingular_matrix(mesh,k,8,8,3.,None,destination)
        self.assertIs(actual,destination)
        self.assertTrue(destination.flags.writeable)
        self.assertEqual(state.cache_stats['stores'],0)

    def test_all_zero_first_batch_keeps_factor_diagnostics(self):
        with mock.patch.dict(os.environ,{'GHOST_CPU_RHS_COMPRESSION':'on','GHOST_CPU_FACTORIZATION':'dense'}):
            factor=DenseFactor(np.eye(64,dtype=complex))
            result=solve(factor,np.zeros((64,40),complex),SweepBasis(40))
            np.testing.assert_array_equal(result,0)
            self.assertEqual(factor.event['rhs_batches'],1)
            self.assertEqual(factor.event['max_backward_error'],0)

    def test_shared_basis_extends_and_repairs_only_failed_columns(self):
        rng = np.random.RandomState(44)
        n = 96
        a = np.eye(n)*4+(.01j*rng.randn(n,n))
        b = rng.randn(n,3) @ (rng.randn(3,64)+1j*rng.randn(3,64))
        with mock.patch.dict(os.environ, {'GHOST_CPU_RHS_COMPRESSION':'on', 'GHOST_CPU_FACTORIZATION':'dense'}):
            factor = DenseFactor(a)
            basis = SweepBasis(64)
            np.testing.assert_allclose(solve(factor,b,basis), np.linalg.solve(a,b), rtol=2e-12, atol=2e-13)
            initial = factor.event['sweep_compression']['solved_columns']
            np.testing.assert_allclose(solve(factor,b[:,::-1],basis), np.linalg.solve(a,b[:,::-1]), rtol=2e-12, atol=2e-13)
            self.assertEqual(initial,factor.event['sweep_compression']['solved_columns'])
            extension = b + rng.randn(n,2) @ rng.randn(2,64)
            np.testing.assert_allclose(solve(factor,extension,basis), np.linalg.solve(a,extension), rtol=2e-12, atol=2e-13)
            self.assertLessEqual(basis.q.shape[1],5)
            coefficients = rng.randn(basis.q.shape[1],64).astype(complex)
            coefficients[0] = 0
            coefficients[0,5] = 1
            rhs = (basis.q*basis.scale[:,None]) @ coefficients
            rhs[:,7] = 0
            basis.x[:,0] += .001
            actual = solve(factor,rhs,basis)
            np.testing.assert_allclose(actual,np.linalg.solve(a,rhs),rtol=2e-12,atol=2e-13)
            np.testing.assert_array_equal(actual[:,7],0)
            self.assertEqual(factor.event['sweep_compression']['fallback_columns'],1)
            self.assertLessEqual(basis.q.shape[1],basis.capacity)

    def test_residual_evidence_is_exact_and_not_retained(self):
        rng = np.random.RandomState(710)
        a = np.eye(260)*10 + rng.randn(260,2) @ rng.randn(2,260)*.001j
        b = rng.randn(260,3)+1j*rng.randn(260,3)
        for factor in (HierarchicalFactor(a), RefinedLU(a)):
            for trans, matrix in ((0,a),(1,a.T),(2,a.conj().T)):
                x,residual = factor.solve(b,trans,return_residual=True)
                np.testing.assert_allclose(residual,b-matrix@x,atol=3e-14,rtol=0)
                np.testing.assert_allclose(x,np.linalg.solve(matrix,b),atol=2e-12,rtol=2e-12)
                ref = weakref.ref(residual)
                del x,residual
                self.assertIsNone(ref())
        for mode,precision in (('hierarchical','double'),('dense','mixed')):
            with mock.patch.dict(os.environ,{'GHOST_CPU_FACTORIZATION':mode}),linear_precision(precision):
                factor = DenseFactor(a)
                x = factor.solve(b)
                np.testing.assert_allclose(x,np.linalg.solve(a,b),atol=2e-12,rtol=2e-12)
                self.assertEqual(factor.event['residual_products_reused'],1)

    def test_operator_destinations_preserve_strided_storage_and_accumulation(self):
        mesh,infos,k = prepared('pec','TE',20)
        expected = ops._assemble_linear_operator_matrices_multi(mesh,k,True,[None])[0]
        n = len(mesh.nodes)
        storage = np.ones((2*n,2*n),complex,order='F')
        s,kmat = ops._assemble_linear_operator_matrices(mesh,k,True,
            single_layer_destination=storage[:n,n:],double_layer_destination=storage[n:,:n])
        self.assertTrue(np.shares_memory(s,storage))
        self.assertTrue(np.shares_memory(kmat,storage))
        np.testing.assert_allclose(s,expected[0]+1,rtol=2e-13,atol=2e-15)
        np.testing.assert_allclose(kmat,expected[1]+1,rtol=2e-13,atol=2e-15)
        readonly = np.zeros((n,n),complex)
        readonly.flags.writeable = False
        with self.assertRaises(ValueError):
            ops._assemble_linear_operator_matrices(mesh,k,True,single_layer_destination=readonly)

    def test_partial_joint_table_uses_exact_out_of_domain_values(self):
        mesh,_,_ = prepared('pec','TE',20)
        k = 30-4000j
        state = execution.CPUState()
        with execution._STATE.override(state):
            fg,fh = kernels.select_far_kernels(mesh,k,ops._far_green_into,ops._far_hankel1_into)
            self.assertTrue(state.table_events[0]['partial_domain'])
            self.assertTrue(hasattr(fg,'pair'))
            distances = np.array([[.001,.003,.07,.09]])
            g,h = np.empty(distances.shape,complex),np.empty(distances.shape,complex)
            fg.pair(k,False,distances,abs(k)*distances,np.empty_like(distances),g,h)
            reference = kernels.values(k,distances)
            np.testing.assert_allclose(g,reference[...,0],rtol=2e-13,atol=1e-280)
            np.testing.assert_allclose(h,reference[...,1],rtol=2e-13,atol=1e-280)
            self.assertEqual(state.table_bytes,sum(t.evidence['bytes'] for t in state.tables.values() if t is not None))

    def test_norm_layouts_match_numpy_with_bounded_blocks(self):
        rng=np.random.RandomState(421)
        a=rng.randn(129,63)+1j*rng.randn(129,63)
        for array in (a,np.asfortranarray(a),a.T,a[::2,::3]):
            self.assertAlmostEqual(matrix_inf_norm(array,256),np.linalg.norm(array,np.inf),places=11)


if __name__ == '__main__':
    unittest.main()
