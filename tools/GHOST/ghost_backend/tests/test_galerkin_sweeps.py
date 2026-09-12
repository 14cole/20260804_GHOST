"""Full-angle sweep acceleration preserves the original discrete equations."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.execution.options import execution_scope
from ghost_backend.linalg.dense import DenseFactor
from ghost_backend.linalg.sweep import SweepBasis, solve


class GalerkinSweepTests(unittest.TestCase):
    def setUp(self):
        scope = execution_scope(dict(factorization='dense', rhs_compression='on', blas_threads=1), limit_blas=True)
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        self.rng = np.random.RandomState(20260912)
        self.n = 160
        self.a = np.eye(self.n)*4 + .001j*self.rng.randn(self.n, self.n)

    def assert_solution(self, factor, rhs, result):
        np.testing.assert_allclose(result, np.linalg.solve(self.a, rhs), rtol=2e-12, atol=2e-13)
        residual = self.a @ result - rhs
        denominator = factor.matrix_inf*np.max(abs(result), axis=0)+np.max(abs(rhs), axis=0)
        self.assertLess(np.max(np.max(abs(residual), axis=0)/np.maximum(denominator, 1e-300)), 1e-12)

    def test_full_181_and_1801_direction_reconstruction(self):
        directions = self.rng.randn(self.n, 5)+1j*self.rng.randn(self.n, 5)
        for count in (181, 1801):
            angles = np.linspace(0, np.pi, count)
            rhs = directions @ np.exp(1j*np.arange(5)[:, None]*angles)
            original = rhs.copy()
            factor, basis = DenseFactor(self.a), SweepBasis(256)
            chunks = [solve(factor, rhs[:, start:start+256], basis) for start in range(0, count, 256)]
            self.assert_solution(factor, rhs, np.column_stack(chunks))
            np.testing.assert_array_equal(rhs, original)
            self.assertEqual(factor.event['factorizations'], 1)
            self.assertLess(factor.event['sweep_compression']['solved_columns'], count//2)
            if count == 1801:
                self.assertGreater(factor.event['sweep_compression'].get('sampled_basis_accepts', 0), 0)
            self.assertLessEqual(basis.q.shape[1], basis.capacity)

    def test_unsampled_illumination_is_checked_before_acceptance(self):
        rhs = self.rng.randn(self.n, 3) @ self.rng.randn(3, 256)
        rhs = rhs.astype(complex)
        sampled = set(np.linspace(0, 255, 64).astype(int))
        isolated = next(i for i in range(256) if i not in sampled)
        rhs[:, isolated] += self.rng.randn(self.n)
        factor = DenseFactor(self.a)
        result = solve(factor, rhs, SweepBasis(256))
        self.assert_solution(factor, rhs, result)
        self.assertEqual(factor.event['sweep_compression'].get('sampled_basis_accepts', 0), 0)
        self.assertEqual(factor.event['sweep_compression']['solved_columns'], 4)
        # A failed proposal must not impose the same overhead on every batch.
        self.assert_solution(factor, rhs, solve(factor, rhs, SweepBasis(256)))
        self.assertEqual(factor.event['sweep_compression']['sampled_basis_attempts'], 1)

    def test_basis_refresh_preserves_capacity_factor_and_accuracy(self):
        factor, basis = DenseFactor(self.a), SweepBasis(12)
        # Distinct local illumination subspaces exhaust the initial retained
        # span; a fresh local span still saves solves in subsequent batches.
        for _ in range(5):
            rhs = self.rng.randn(self.n, 6) @ (self.rng.randn(6, 128)+1j*self.rng.randn(6, 128))
            result = solve(factor, rhs, basis)
            self.assert_solution(factor, rhs, result)
            self.assertLessEqual(basis.q.shape[1], 12)
            self.assertEqual(basis.q.shape, basis.x.shape)
        event = factor.event['sweep_compression']
        self.assertGreater(event.get('basis_restarts', 0), 0)
        self.assertLessEqual(event['solved_columns'], 35)
        self.assertEqual(event['fallback_batches'], 0)
        self.assertEqual(event['input_columns'], 640)
        self.assertEqual(factor.event['factorizations'], 1)

    def test_uncompressible_batch_uses_original_illuminations(self):
        rhs = self.rng.randn(self.n, 181)+1j*self.rng.randn(self.n, 181)
        factor = DenseFactor(self.a)
        self.assert_solution(factor, rhs, solve(factor, rhs, SweepBasis(256)))
        self.assertEqual(factor.event['sweep_compression']['fallback_batches'], 1)
        self.assertEqual(factor.event['sweep_compression']['solved_columns'], 181)

    def test_cancel_during_proposal_does_not_publish_or_solve(self):
        rhs = self.rng.randn(self.n, 3) @ self.rng.randn(3, 256)
        factor = DenseFactor(self.a)
        calls = []
        def checkpoint():
            calls.append(1)
            if len(calls) == 2:
                raise InterruptedError('cancelled')
        factor.checkpoint = checkpoint
        with self.assertRaises(InterruptedError):
            solve(factor, rhs, SweepBasis(256))
        self.assertEqual(factor.event['rhs_batches'], 0)

    def test_sampled_reconstruction_keeps_compressed_operator_error_checks(self):
        from test_compressed_path import Exact
        from ghost_backend.compressed.operator import StreamedOperator
        from ghost_backend.compressed.factor import CompressedFactor
        operator = StreamedOperator(Exact(self.a), np.arange(self.n)[:, None], tile=40)
        factor = CompressedFactor(operator)
        rhs = self.rng.randn(self.n, 3) @ (self.rng.randn(3, 256)+1j*self.rng.randn(3, 256))
        self.assert_solution(factor, rhs, solve(factor, rhs, SweepBasis(256)))
        self.assertGreater(factor.event['sweep_compression'].get('sampled_basis_accepts', 0), 0)


class PhysicalSweepTests(unittest.TestCase):
    def test_galerkin_fields_match_uncompressed_solves_for_full_sweeps(self):
        from ghost_backend.twod import solver
        from test_experimental_cpu import fixture, fields
        from test_thin_sheet import sheet_snapshot
        cases = [fixture(kind, 32) for kind in ('pec', 'ibc', 'lossy', 'layered', 'mixed')]
        sheet = sheet_snapshot([[-.05, 0], [.05, 0]], panels=32)
        cases += [sheet, dict(sheet, ibcs=[['1', 'thin_dielectric', '.0005', '2']],
                              dielectrics=[['2', '3', '-.02', '1.4', '-.01']])]
        for index, snapshot in enumerate(cases):
            for count in (181, 1801):
                with self.subTest(case=index, angles=count):
                    results = []
                    for compression in ('off', 'on'):
                        results.append(solver.solve_monostatic_rcs_2d_survey(
                            snapshot, [.6], np.linspace(0, 180, count).tolist(), geometry_units='meters',
                            solver_method='experimental_cpu', execution_options=dict(
                                factorization='dense', rhs_compression=compression, blas_threads=1)))
                    for pol in ('VV', 'HH'):
                        expected, actual = (fields(result, pol) for result in results)
                        peak = max(float(np.max(abs(expected))), 1e-280)
                        self.assertLess(float(np.max(abs(actual-expected)))/peak, 1e-10)
                    self.assertEqual(results[1]['metadata']['dense_factorization_count'], 2)


if __name__ == '__main__':
    unittest.main()
