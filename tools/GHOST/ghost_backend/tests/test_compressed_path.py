"""Compressed coefficient/error contracts and public certification integration."""
import os,sys,unittest,weakref,tempfile
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ghost_backend.twod.solver as rcs
from ghost_backend.compressed.operator import StreamedOperator
from ghost_backend.compressed.factor import CompressedFactor
from ghost_backend.compressed.polarization_cache import SpooledOperator
from ghost_backend.linalg.hierarchical import HierarchicalRejected
from ghost_backend.linalg.sweep import SweepBasis, solve as solve_sweep
from test_experimental_cpu import fixture


class Exact:
    dropped_routes=0
    def __init__(self,a):self.a=a;self.n=len(a);self.calls=self.entries=0
    def get_with_error(self,rows,cols):
        value=self.a[np.ix_(rows,cols)].copy();self.calls+=1;self.entries+=value.size
        return value,np.zeros(value.shape)


class CompressedPathTests(unittest.TestCase):
    def setUp(self):
        self.environment=mock.patch.dict(os.environ,{'GHOST_CPU_FACTORIZATION':'compressed','GHOST_DENSE_BACKEND':'cpu'})
        self.environment.start();self.addCleanup(self.environment.stop)

    def system(self,n=260):
        rng=np.random.RandomState(17);u=rng.randn(n,5)+1j*rng.randn(n,5);v=rng.randn(5,n)+1j*rng.randn(5,n)
        a=np.eye(n)*20+u@v/n
        a=np.logspace(-2,2,n)[:,None]*a*np.logspace(1,-1,n)[None,:]
        source=Exact(a);operator=StreamedOperator(source,np.arange(n)[:,None],tile=64)
        return a,source,operator

    def test_operator_adjoint_error_bounds_and_equilibration(self):
        a,source,operator=self.system();b=np.arange(len(a)*2).reshape(len(a),2)+1j
        for trans,expected in ((0,a),(1,a.T),(2,a.conj().T)):
            actual=operator.matmul(b,trans)
            np.testing.assert_allclose(actual,expected@b,rtol=2e-13,atol=1e-9)
            axis_error=operator.row_error if trans==0 else operator.column_error
            self.assertLess(np.max(abs(actual-expected@b)),np.max(axis_error)*np.max(abs(b))+1e-8)
        expected=rcs._equilibrated_scaling_and_norm_1(a)
        actual=operator.equilibrate()
        for x,y in zip(actual,expected):np.testing.assert_allclose(x,y,rtol=3e-13,atol=1e-14)
        refs=[weakref.ref(source),weakref.ref(a)];del source,a
        self.assertTrue(all(ref() is None for ref in refs))

    def test_inverse_transposes_conditions_and_physical_sweep(self):
        a,source,operator=self.system();diagnostics={};factor=CompressedFactor(operator,diagnostics)
        rng=np.random.RandomState(74);b=rng.randn(len(a),3)+1j*rng.randn(len(a),3)
        for trans,expected in ((0,a),(1,a.T),(2,a.conj().T)):
            np.testing.assert_allclose(factor.inverse(b,trans),np.linalg.solve(expected,b),rtol=1e-10,atol=1e-11)
        self.assertTrue(np.isfinite(diagnostics['condition_est']))
        basis=SweepBasis(64);rhs=b@(rng.randn(3,64)+1j*rng.randn(3,64))
        np.testing.assert_allclose(solve_sweep(factor,rhs,basis),np.linalg.solve(a,rhs),rtol=1e-10,atol=2e-11)
        operator.row_error.fill(1e9)
        with self.assertRaises((HierarchicalRejected,RuntimeError)):solve_sweep(factor,rhs,basis)

    def test_storage_cancellation_and_invalid_rhs_reject(self):
        a,source,operator=self.system(32)
        with mock.patch('ghost_backend.compressed.runtime.storage_budget',return_value=operator.bytes+1):
            with self.assertRaises(MemoryError):CompressedFactor(operator)
        def cancel():raise InterruptedError('canceled')
        with self.assertRaises(InterruptedError):CompressedFactor(operator,checkpoint=cancel)
        factor=CompressedFactor(operator)
        for b in (np.ones(31),np.ones((32,0)),np.full(32,np.nan)):
            with self.assertRaises(ValueError):factor.inverse(b)
        with mock.patch.dict(os.environ,{'GHOST_COMPRESSED_STORAGE_MIB':'NaN'}):
            with self.assertRaises(ValueError):CompressedFactor(operator)

    def test_spool_is_removed_and_requires_load(self):
        a,source,operator=self.system(32)
        with tempfile.TemporaryDirectory() as directory:
            disk=SpooledOperator(source,np.arange(32)[:,None],tile=16,directory=directory)
            path=disk.path
            with self.assertRaises(ValueError):disk.matmul(np.ones(32))
            disk.load();self.assertFalse(path.exists())
            np.testing.assert_allclose(disk.matmul(np.ones(32)),a@np.ones(32),rtol=2e-13)
            corrupt=SpooledOperator(source,np.arange(32)[:,None],tile=16,directory=directory)
            corrupt.file.seek(0);corrupt.file.write(b'bad data');corrupt.file.flush()
            with self.assertRaisesRegex(IOError,'checksum'):corrupt.load()
            self.assertFalse(corrupt.path.exists())
            corrupt.close();self.assertFalse(corrupt.path.exists())

    def test_bounded_gmres_repairs_stalled_corrections(self):
        a=np.diag(np.linspace(1,4,16)).astype(complex)
        op=StreamedOperator(Exact(a),np.arange(16)[:,None],tile=8)
        factor=CompressedFactor(op)
        with mock.patch.object(factor.factor,'apply',side_effect=lambda b,**kw:b.copy()):
            actual=factor.inverse(np.arange(1,17).astype(complex))
        np.testing.assert_allclose(actual,np.linalg.solve(a,np.arange(1,17)),rtol=1e-11,atol=1e-12)
        self.assertGreater(factor.event['gmres_columns'],0)

    def test_public_mesh_certificate_and_no_global_dense_factor(self):
        value=fixture('pec',64)
        with mock.patch('ghost_backend.linalg.dense.DenseFactor.__init__',side_effect=AssertionError('dense factor')):
            result=rcs.solve_monostatic_rcs_2d_certified(value,[.6],[0.,30.,90.,180.,360.],
                solver_method='experimental_cpu',max_panels=10000)
        self.assertTrue(result['metadata']['mesh_convergence_certified'])
        self.assertEqual(result['metadata']['solver_method'],'compressed_experimental_cpu')
        self.assertTrue(result['metadata']['compressed_factors'])
        self.assertGreater(result['metadata']['panel_count'],64)

    def test_boundary_density_diagnostics_match_dense_without_dense_factor_or_farfield(self):
        for kind in ('pec', 'ibc', 'lossy', 'magnetic', 'coated', 'layered', 'mixed'):
            for pol in ('TE', 'TM'):
                with self.subTest(kind=kind, polarization=pol):
                    args = (fixture(kind, 32), .6, 17.3, pol)
                    with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION': 'dense'}):
                        expected = rcs.compute_boundary_densities(*args, geometry_units='meters')
                    with mock.patch('ghost_backend.linalg.dense.DenseFactor.__init__', side_effect=AssertionError('dense factor')), \
                         mock.patch.object(rcs, '_farfield_linear_density_many', side_effect=AssertionError('unused far field')):
                        actual = rcs.compute_boundary_densities(*args, geometry_units='meters')
                    fields = lambda r: np.asarray(r['density_real']) + 1j*np.asarray(r['density_imag'])
                    a, b = fields(actual), fields(expected)
                    self.assertLess(np.max(abs(a-b))/max(np.max(abs(b)), 1e-280), 1e-10)
                    self.assertEqual(actual['formulation'], expected['formulation'])

    def test_boundary_density_cancellation_reaches_compressed_assembly(self):
        import threading
        from ghost_backend.compressed.coefficients import NativeOracle
        from ghost_backend.twod.assembly.session import current_session
        event = threading.Event()
        original = NativeOracle.get_with_error
        def cancel_after_query(oracle, *args, **kwargs):
            result = original(oracle, *args, **kwargs)
            event.set()
            return result
        with mock.patch.object(NativeOracle, 'get_with_error', cancel_after_query), \
             mock.patch('ghost_backend.compressed.factor.CompressedFactor.__init__', side_effect=AssertionError('factor after cancellation')):
            with self.assertRaises(InterruptedError):
                rcs.compute_boundary_densities(fixture('pec', 32), .6, 0., 'TE',
                    geometry_units='meters', abort_event=event)
        self.assertIsNone(current_session())

    def test_failed_quality_gate_removes_pending_polarization(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch('ghost_backend.compressed.runtime.tempfile.gettempdir',return_value=directory):
            with self.assertRaisesRegex(ValueError,'Quality gate failed'):
                rcs.solve_monostatic_rcs_2d_certified(fixture('pec',64),[.6],[0.,90.],
                    solver_method='experimental_cpu',quality_thresholds={'condition_est_max':1.01})
            self.assertFalse(list(Path(directory).glob('ghost-tm-*.bin')))

    def test_bistatic_and_multiple_frequency_grids(self):
        for kind in ('pec','lossy','mixed'):
            value=fixture(kind,24)
            actual=rcs.solve_bistatic_rcs_2d(value,[.6],[0.,60.],[10.,90.,180.],
                strict_quality_gate=True,compute_condition_number=True)
            with mock.patch.dict(os.environ,{'GHOST_CPU_FACTORIZATION':'dense'}):
                expected=rcs.solve_bistatic_rcs_2d(value,[.6],[0.,60.],[10.,90.,180.],
                    strict_quality_gate=True,compute_condition_number=True)
            for pol in ('VV','HH'):
                fields=lambda r:np.asarray([complex(x['rcs_amp_real'],x['rcs_amp_imag']) for x in r['co_solved_samples'][pol]])
                np.testing.assert_allclose(fields(actual),fields(expected),rtol=2e-10,atol=1e-12)
        result=rcs.solve_monostatic_rcs_2d(fixture('pec',24),[.5,.6],[0.,180.,360.],
            solver_method='experimental_cpu',compute_condition_number=True)
        self.assertEqual(len(result['metadata']['frequency_metadata']),2)
        self.assertEqual(result['metadata']['dense_factorization_count'],4)

    def test_budget_changes_invalidate_provenance(self):
        from ghost_backend.execution.provenance import runtime_environment_fingerprint
        before=runtime_environment_fingerprint()
        with mock.patch.dict(os.environ,{'GHOST_COMPRESSED_STORAGE_MIB':'32'}):
            self.assertNotEqual(before,runtime_environment_fingerprint())

    def test_more_than_512_angles_stream_and_match_dense(self):
        angles=np.linspace(-15.,375.,1027).tolist()
        value=fixture('pec',32)
        actual=rcs.solve_monostatic_rcs_2d(value,[.6],angles,solver_method='experimental_cpu',
            compute_condition_number=True,strict_quality_gate=True)
        with mock.patch.dict(os.environ,{'GHOST_CPU_FACTORIZATION':'dense'}):
            expected=rcs.solve_monostatic_rcs_2d(value,[.6],angles,solver_method='experimental_cpu',
                compute_condition_number=True,strict_quality_gate=True)
        for pol in ('VV','HH'):
            fields=lambda r:np.asarray([complex(x['rcs_amp_real'],x['rcs_amp_imag']) for x in r['co_solved_samples'][pol]])
            self.assertEqual(len(fields(actual)),1027)
            np.testing.assert_allclose(fields(actual),fields(expected),rtol=2e-10,atol=1e-12)
        for event in actual['metadata']['compressed_factors']:
            self.assertLessEqual(event['max_rhs_columns'],256)
            self.assertEqual(event['factorizations'],1)

    def test_loss_envelopes_bound_reference_kernels(self):
        from ghost_backend.compressed.regional_coefficients import hankel_envelopes
        from scipy.special import hankel2
        for k in (2.-.1j,2.-5j,100.-40j):
            distance=np.geomspace(1.01,500.,81)/(-k.imag)
            g,h=hankel_envelopes(k,distance)
            self.assertTrue(np.all(abs(.25j*hankel2(0,k*distance))<=g))
            self.assertTrue(np.all(abs(.25j*k*hankel2(1,k*distance))<=h))
        for k in (1.,1.+1j,-1.-1j):
            self.assertIsNone(hankel_envelopes(k,np.array([100.])))
        self.assertIsNone(hankel_envelopes(1.-1j,np.array([.5])))

    def test_pair_failure_and_interrupted_load_remove_owned_spools(self):
        from ghost_backend.compressed.polarization_cache import build_pair
        a,source,operator=self.system(32)
        class Pair:
            oracles=(source,source)
            def get_with_error(self,rows,cols):
                return [source.get_with_error(rows,cols) for _ in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(MemoryError):
                build_pair(Pair(),np.arange(32)[:,None],tile=16,budget=4000,spool_directory=directory)
            self.assertFalse(list(Path(directory).glob('ghost-tm-*.bin')))
            for reason in ('truncated','canceled'):
                disk=SpooledOperator(source,np.arange(32)[:,None],tile=16,directory=directory)
                try:
                    if reason=='truncated':
                        disk.file.truncate(0)
                        with self.assertRaisesRegex(IOError,'Truncated'):disk.load()
                    else:
                        def cancel():raise InterruptedError('canceled')
                        disk.checkpoint=cancel
                        with self.assertRaises(InterruptedError):disk.load()
                    self.assertFalse(disk.path.exists())
                    with self.assertRaises(ValueError):disk.matmul(np.ones(32))
                finally:disk.close()
                self.assertFalse(disk.path.exists())

    def test_spool_close_error_still_removes_file(self):
        a,source,operator=self.system(16)
        with tempfile.TemporaryDirectory() as directory:
            disk=SpooledOperator(source,np.arange(16)[:,None],tile=8,directory=directory)
            original=disk.file
            class FailedFlush:
                def close(self):
                    original.close()
                    raise OSError('simulated disk flush failure')
            disk.file=FailedFlush()
            with self.assertRaisesRegex(OSError,'flush'):disk.close()
            self.assertFalse(disk.path.exists())
            self.assertIsNone(disk.file)

    def test_mixed_precision_is_rejected_before_assembly(self):
        from ghost_backend.linalg.refined_lu import linear_precision
        with linear_precision('mixed'), mock.patch('ghost_backend.compressed.runtime.regional',side_effect=AssertionError('assembled')):
            with self.assertRaisesRegex(ValueError,'double precision'):
                rcs.solve_monostatic_rcs_2d(fixture('layered',24),[.6],[0.],compute_condition_number=True)
        a,source,operator=self.system(16)
        with linear_precision('mixed'):
            with self.assertRaisesRegex(ValueError,'double precision'):CompressedFactor(operator)

    def test_large_inverse_block_checks_respect_query_workspace_cap(self):
        from ghost_backend.compressed.inverse import Block
        class BoundedOracle:
            def get(self,rows,cols):
                if len(rows)*len(cols)*16>16*1024**2:
                    raise MemoryError('oversized query')
                return np.ones((len(rows),len(cols)),complex)
        rows=np.arange(65);cols=np.arange(33000)
        block=Block(BoundedOracle(),rows,cols,lambda:None)
        error,pivot=block.error(np.ones((65,1),complex),np.ones((1,33000),complex))
        self.assertEqual(error,0.)

    def test_singular_system_rejects_without_dense_global_fallback(self):
        operator=StreamedOperator(Exact(np.zeros((16,16),complex)),np.arange(16)[:,None],tile=8)
        with mock.patch('ghost_backend.linalg.dense.DenseFactor.__init__',side_effect=AssertionError('dense factor')):
            with self.assertRaises(Warning):CompressedFactor(operator)


if __name__=='__main__':unittest.main()
