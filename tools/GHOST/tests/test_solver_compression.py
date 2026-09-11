"""Independent matrix/field equivalence and compressed-solve rejection checks."""
import os
import sys
from pathlib import Path
import unittest
from unittest import mock
import weakref
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Backend'))
import ghost_backend.twod.solver as rcs
import ghost_backend.twod.operators as ops
import ghost_backend.twod.formulations.regions as multi_region
import ghost_backend.linalg.hierarchical as hf
from ghost_backend.linalg.dense import DenseFactor
from ghost_backend.linalg.sweep import solve as compress_sweep
from test_compact_multi_region import prepared
from test_experimental_cpu import fixture, solve, fields
from test_thin_sheet import sheet_mesh
from ghost_backend.twod.formulations.thin_layer import solve_thin_layer_fields


class SolverCompressionTests(unittest.TestCase):
    def test_uncompressible_block_and_cancellation_stop_construction(self):
        rng = np.random.RandomState(881)
        a = rng.randn(64, 64)+1j*rng.randn(64, 64)
        block = hf.Block(a, np.arange(64), np.arange(64), lambda: None)
        with self.assertRaises(hf.HierarchicalRejected):
            hf.compress(block, maximum_rank=8)
        def cancel():
            raise InterruptedError('canceled')
        with self.assertRaises(InterruptedError):
            hf.HierarchicalFactor(a, checkpoint=cancel)

    def test_compression_settings_are_part_of_runtime_provenance(self):
        from ghost_backend.execution.provenance import (
            runtime_environment_payload,
            runtime_environment_fingerprint,
        )
        with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION':'dense', 'GHOST_CPU_RHS_COMPRESSION':'off'}):
            reference = runtime_environment_fingerprint()
        with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION':'hierarchical', 'GHOST_CPU_RHS_COMPRESSION':'on'}):
            self.assertNotEqual(reference, runtime_environment_fingerprint())
            self.assertEqual(runtime_environment_payload()['cpu_factorization'], 'hierarchical')

    def test_direct_scatter_matches_independent_operator_system(self):
        for kind in ('layered', 'coated', 'mixed', 'equal_k'):
            for pol in ('TE', 'TM'):
                for orders in ((4, 4), (4, 8)):
                    with self.subTest(kind=kind, pol=pol, orders=orders):
                        mesh, infos, _ = prepared(kind, pol, 24)
                        expected, _ = multi_region._assemble_system_fresh(mesh, infos, pol, *orders)
                        with mock.patch.object(ops, '_ASSEMBLY_TILE', 13), mock.patch.object(ops, '_ASSEMBLY_THREADS', 2):
                            actual, _ = multi_region.assemble_system(mesh, infos, pol, *orders)
                        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-17)

    def test_hierarchical_inverse_and_adjoint_match_dense(self):
        rng = np.random.RandomState(119)
        n = 513
        u = rng.randn(n, 5)+1j*rng.randn(n, 5)
        v = rng.randn(5, n)+1j*rng.randn(5, n)
        a = np.eye(n)*30 + u @ v / n
        b = rng.randn(n, 3)+1j*rng.randn(n, 3)
        factor = hf.HierarchicalFactor(a)
        self.assertLess(factor.bytes, a.nbytes*.65)
        self.assertGreater(factor.evidence['low_rank_blocks'], 0)
        for trans, op in ((0, a), (1, a.T), (2, a.conj().T)):
            np.testing.assert_allclose(factor.solve(b, trans), np.linalg.solve(op, b), rtol=2e-12, atol=2e-14)
        np.testing.assert_allclose(factor.solve(b[:, 0]), np.linalg.solve(a, b[:, 0]), rtol=2e-12, atol=2e-14)

    def test_coarse_inverse_rebuild_releases_tree_and_preserves_exact_gate(self):
        a = np.eye(260, dtype=complex)*4
        factor = hf.HierarchicalFactor(a)
        self.assertEqual(factor.tolerance, 1e-6)
        original = factor._apply
        old = weakref.ref(factor.root)
        def stalled(b, trans):
            if factor.tolerance > 2e-10:
                return np.zeros_like(b)
            self.assertIsNone(old())
            return original(b, trans)
        with mock.patch.object(factor, '_apply', side_effect=stalled):
            x, residual = factor.solve(np.ones((260, 2)), return_residual=True)
        np.testing.assert_array_equal(x, np.full((260, 2), .25))
        np.testing.assert_array_equal(residual, np.zeros((260, 2)))
        self.assertEqual(factor.evidence['builds'], 2)
        self.assertEqual(factor.evidence['tighter_rebuilds'], 1)
        with mock.patch.object(factor, '_apply', side_effect=lambda b, trans:np.zeros_like(b)) as stalled:
            with self.assertRaises(hf.HierarchicalRejected):factor.solve(np.ones(260))
        self.assertEqual(stalled.call_count, 8)  # Initial inverse plus seven useful corrections.
        self.assertEqual(factor.evidence['builds'], 2)

    def test_coarse_construction_rejection_releases_work_before_retry(self):
        original = hf.HierarchicalFactor._build
        refs=[]
        def trial(factor, ids):
            if factor.tolerance > 2e-10:
                buffer=np.ones(1024, complex);refs.append(weakref.ref(buffer))
                raise hf.HierarchicalRejected('coarse inverse singular')
            self.assertIsNone(refs[0]())
            return original(factor, ids)
        with mock.patch.object(hf.HierarchicalFactor, '_build', trial), \
             mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION':'hierarchical'}):
            factor=DenseFactor(np.eye(260, dtype=complex))
            np.testing.assert_array_equal(factor.solve(np.ones(260)),np.ones(260))
            self.assertEqual(factor.event['factorizations'],2)

    def test_hierarchical_input_validation_does_not_trigger_rebuild(self):
        factor=hf.HierarchicalFactor(np.eye(8, dtype=complex))
        for rhs in (np.ones(7),np.ones((8,0)),np.full(8,np.nan),np.ones((8,2,2))):
            with self.assertRaises(ValueError):factor.solve(rhs)
        self.assertEqual(factor.evidence['builds'],1)

    def test_compressed_and_full_complex_fields_all_boundary_families(self):
        for kind in ('pec', 'ibc', 'lossy', 'magnetic', 'layered', 'coated', 'mixed'):
            with self.subTest(kind=kind):
                snapshot = fixture(kind, 24)
                angles = np.linspace(0, 360, 73)
                with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION':'dense', 'GHOST_CPU_RHS_COMPRESSION':'off'}):
                    expected = solve(snapshot, angles=angles)
                with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION':'hierarchical', 'GHOST_CPU_RHS_COMPRESSION':'on'}):
                    actual = solve(snapshot, angles=angles)
                for pol in ('VV', 'HH'):
                    np.testing.assert_allclose(fields(actual, pol), fields(expected, pol), rtol=2e-9, atol=1e-12)
                    self.assertEqual(actual['metadata']['channel_metadata'][pol]['linear_backend'], 'cpu_hierarchical')

    def test_sweep_rank_reconstruction_and_qr_failure_fallback(self):
        rng = np.random.RandomState(771)
        n = 160
        a = np.eye(n)*4 + rng.randn(n, n)*.01
        b = rng.randn(n, 7) @ (rng.randn(7, 64)+1j*rng.randn(7, 64))
        with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION':'dense', 'GHOST_CPU_RHS_COMPRESSION':'on'}):
            factor = DenseFactor(a)
            expected = np.linalg.solve(a, b)
            np.testing.assert_allclose(compress_sweep(factor, b), expected, rtol=3e-12, atol=2e-14)
            event = factor.event['sweep_compression']
            self.assertEqual(event['accepted_batches'], 1)
            self.assertLessEqual(event['solved_columns'], 8)
            with mock.patch('ghost_backend.linalg.sweep.la.qr', side_effect=np.linalg.LinAlgError('QR stalled')):
                np.testing.assert_allclose(compress_sweep(factor, b), expected, rtol=3e-12, atol=2e-14)
            self.assertEqual(event['fallback_batches'], 1)

    def test_rejected_hierarchy_is_freed_before_dense_fallback(self):
        original_lu = rcs._SCIPY_LINALG.lu_factor
        for stage in ('build', 'condition', 'solve'):
            refs = []
            class Rejected:
                def __init__(self, a, *args):
                    self.buffer = np.ones(1024, complex)
                    self.evidence = {}
                    refs.append(weakref.ref(self.buffer))
                    if stage == 'build':
                        raise hf.HierarchicalRejected('rank budget')
                def solve(self, *args, **kwargs):
                    raise hf.HierarchicalRejected('residual stalled')
            def checked_lu(*args, **kwargs):
                self.assertIsNone(refs[0]())
                return original_lu(*args, **kwargs)
            with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION':'auto'}), \
                 mock.patch.object(hf, 'HierarchicalFactor', Rejected), \
                 mock.patch.object(rcs._SCIPY_LINALG, 'lu_factor', checked_lu):
                factor = DenseFactor(np.eye(2048, dtype=complex), {} if stage == 'condition' else None)
                actual = factor.solve(np.ones((2048, 2)))
                np.testing.assert_array_equal(actual, np.ones((2048, 2)))
                self.assertIn('fell back', factor.fallback_reason)

    def test_strict_hierarchy_rejects_without_full_lu(self):
        with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION':'hierarchical'}), \
             mock.patch.object(hf, 'HierarchicalFactor', side_effect=hf.HierarchicalRejected('rank budget')), \
             mock.patch.object(rcs._SCIPY_LINALG, 'lu_factor', side_effect=AssertionError('full LU')):
            with self.assertRaisesRegex(RuntimeError, 'rank budget'):
                DenseFactor(np.eye(32, dtype=complex))

    def test_memory_estimate_preserves_auto_fallback_and_strict_savings(self):
        args = dict(nnodes=10000, use_cfie=False, system_dofs=16000, n_rhs=361,
                    dense_resources=dict(formulation='multi_region', operator_entries=10**8,
                        assembly_operator_entries=0, operator_map_bytes=10**7,
                        mass_workspace_bytes=10**7, block_workspace_bytes=10**7,
                        assembly_workspace_bytes=10**8))
        with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION':'dense'}):
            dense = rcs._estimate_memory_gb(**args)
        with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION':'auto'}):
            self.assertEqual(dense, rcs._estimate_memory_gb(**args))
        with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION':'hierarchical'}):
            self.assertLess(rcs._estimate_memory_gb(**args), dense)

    def test_zero_contrast_layer_skips_operators_and_factorization(self):
        mesh = sheet_mesh([[-.05, 0], [.05, 0]], panels=16)
        with mock.patch.object(rcs, '_assemble_linear_operator_matrices', side_effect=AssertionError('unused S/K')), \
             mock.patch.object(rcs._SCIPY_LINALG, 'lu_factor', side_effect=AssertionError('unused LU')):
            for pol in ('TE', 'TM'):
                diagnostics = {}
                _, field, residual, evidence = solve_thin_layer_fields(mesh, 20., [0., 90.], pol,
                    1., 1., .0001, observation_angles_deg=[0., 120., 240.], condition_diagnostics=diagnostics)
                np.testing.assert_array_equal(field, np.zeros((2, 3)))
                self.assertEqual(evidence['unknowns'], 0)
                self.assertEqual(diagnostics['condition_est'], 1.)


if __name__ == '__main__':
    unittest.main()
