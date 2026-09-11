"""Allocation lifetime, coefficient reuse and sweep contracts for all materials."""
import gc
import os
from pathlib import Path
import sys
import threading
import unittest
import weakref
from unittest import mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'Backend'))
import ghost_backend.twod.solver as rcs
import ghost_backend.twod.formulations.regions as multi_region
import ghost_backend.twod.assembly.session as sessions
from ghost_backend.linalg.dense import DenseFactor
from ghost_backend.twod.assembly.mass import sparse_mass
from test_compact_multi_region import prepared
from test_experimental_cpu import fixture, solve, fields
from test_thin_sheet import sheet_snapshot
from ghost_backend.execution.runtime import replace


class MatrixPipelineTests(unittest.TestCase):
    def test_reused_multi_coefficients_match_fresh_assembly(self):
        for kind in ('layered', 'equal_k', 'coated', 'mixed'):
            for order in (4, 8):
                with self.subTest(kind=kind, order=order):
                    mesh, te, k0 = prepared(kind, 'TE', 24)
                    _, tm, _ = prepared(kind, 'TM', 24)
                    expected, _ = multi_region.assemble_system(mesh, tm, 'TM', order, order)
                    session = sessions.AssemblySession()
                    with sessions._SESSION.override(session):
                        first, _ = multi_region.assemble_system(mesh, te, 'TE', order, order)
                        actual, _ = multi_region.assemble_system(mesh, tm, 'TM', order, order)
                    self.assertIs(first, actual)
                    self.assertEqual(session.reuses, 1)
                    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-16)

    def test_reuse_is_scoped_to_matching_materials_mesh_and_rules(self):
        mesh, infos, _ = prepared('layered', 'TE', 16)
        scenarios = [([replace(info, eps_plus=info.eps_plus*1.1) for info in infos], 8, 8),
                     (infos, 4, 8)]
        for other, obs, src in scenarios:
            session = sessions.AssemblySession()
            with sessions._SESSION.override(session):
                first, _ = multi_region.assemble_system(mesh, infos, 'TE')
                second, _ = multi_region.assemble_system(mesh, other, 'TM', obs, src)
            self.assertIsNot(first, second)
            self.assertEqual(session.reuses, 0)

    def test_cpu_sweep_has_one_factor_per_channel_and_bounded_columns(self):
        snapshots = [fixture(kind, 24) for kind in ('pec','ibc','layered','coated','mixed','lossy')]
        sheet = sheet_snapshot([[-.05, 0], [.05, 0]], panels=24)
        snapshots.append(sheet)
        thin = dict(sheet, ibcs=[['1','thin_dielectric','.0005','2']],
                    dielectrics=[['2','3','-.02','1.4','-.01']])
        snapshots.append(thin)
        angles = np.linspace(0,360,19).tolist()
        for snapshot in snapshots:
            with mock.patch.dict(os.environ, {'GHOST_CPU_ANGLE_BATCH_SIZE':'7'}):
                result = solve(snapshot, angles=angles)
            m = result['metadata']
            self.assertEqual(m['dense_factorization_count'], 2)
            self.assertEqual(m['dense_rhs_batch_count'], 6)
            self.assertEqual(m['dense_rhs_column_count'], 38)
            self.assertEqual(m['dense_max_rhs_columns'], 7)
            with mock.patch.object(sessions, 'current_session', return_value=None):
                reference = solve(snapshot, angles=angles)
            for pol in ('VV', 'HH'):
                np.testing.assert_allclose(fields(result, pol), fields(reference, pol), rtol=3e-10, atol=1e-12)

    def test_sparse_mass_matches_dense_weighted_mass(self):
        mesh, _, _ = prepared('mixed', count=16)
        weights = np.linspace(.2, 2., len(mesh.elements))*(1-.2j)
        np.testing.assert_allclose(sparse_mass(mesh, weights).toarray(),
                                  rcs._assemble_linear_weighted_mass_matrix(mesh, weights), rtol=1e-14, atol=1e-18)

    def test_certification_keeps_both_mesh_checks_and_reports_all_factors(self):
        result = rcs.solve_monostatic_rcs_2d_certified(fixture('lossy', 48), [.6], [0., 90.],
                                                      geometry_units='meters')
        metadata = result['metadata']
        self.assertTrue(metadata['mesh_convergence_certified'])
        self.assertEqual(metadata['dense_factorization_count'], 4)
        self.assertEqual(metadata['assembled_system_reuses'], 2)
        self.assertEqual(metadata['certification_solve_order'], ['base_TE','base_TM','fine_TE','fine_TM'])
        for pol in ('VV', 'HH'):
            gate = metadata['mesh_convergence']['channels'][pol]
            self.assertTrue(gate['passed'])
            self.assertGreater(gate['fine_panel_count'], gate['base_panel_count'])

    def test_default_cpu_cancels_between_batches_and_releases_pending_system(self):
        event = threading.Event()
        original = DenseFactor.solve
        def cancel(factor, rhs):
            result = original(factor, rhs)
            event.set()
            return result
        with mock.patch.dict(os.environ, {'GHOST_CPU_ANGLE_BATCH_SIZE':'7'}), \
             mock.patch.object(DenseFactor, 'solve', cancel):
            with self.assertRaises(InterruptedError):
                rcs.solve_monostatic_rcs_2d(fixture('lossy', 24), [.6], list(range(20)),
                    geometry_units='meters', abort_event=event)
        self.assertIsNone(sessions.current_session())

    def test_multifrequency_bistatic_preserves_grid(self):
        result = rcs.solve_bistatic_rcs_2d(fixture('lossy', 24), [.5, .6], [0., 30.], [11., 90.],
            geometry_units='meters', compute_condition_number=True)
        self.assertEqual(len(result['samples']), 16)
        self.assertEqual(result['metadata']['dense_factorization_count'], 4)

    def test_no_dense_mass_and_no_operator_copies_survive_factorization(self):
        for kind in ('ibc', 'lossy', 'layered', 'mixed'):
            refs, calls = [], []
            original_ops = rcs._assemble_linear_operator_matrices_multi
            original_lu = rcs._SCIPY_LINALG.lu_factor
            def assemble(*args, **kw):
                output = original_ops(*args, **kw)
                for pair in output:
                    for item in pair:
                        if hasattr(item, 'scatter_add'):
                            refs.append(weakref.ref(item))
                            continue
                        array = item.values if hasattr(item, 'values') else item
                        if any(array.strides):
                            refs.append(weakref.ref(array))
                return output
            def factor(matrix, *args, **kw):
                gc.collect()
                calls.append(len(matrix))
                self.assertTrue(all(ref() is None or isinstance(ref(), np.ndarray) and np.shares_memory(ref(), matrix) for ref in refs))
                return original_lu(matrix, *args, **kw)
            with mock.patch.object(rcs, '_assemble_linear_mass_matrix', side_effect=AssertionError('dense mass')), \
                 mock.patch.object(rcs, '_assemble_linear_weighted_mass_matrix', side_effect=AssertionError('dense weighted mass')), \
                 mock.patch.object(rcs, '_assemble_linear_operator_matrices_multi', assemble), \
                 mock.patch.object(rcs._SCIPY_LINALG, 'lu_factor', factor):
                solve(fixture(kind, 24), angles=[0., 90.])
            self.assertEqual(len(calls), 2)

    def test_density_export_skips_far_field_and_uses_checked_solve(self):
        for kind in ('lossy', 'coated', 'ibc'):
            with mock.patch.object(rcs, '_farfield_linear_density_many', side_effect=AssertionError('unused field')), \
                 mock.patch.object(rcs, '_assemble_linear_mass_matrix', side_effect=AssertionError('dense mass')):
                result = rcs.compute_boundary_densities(fixture(kind, 24), .6, 31., 'TM', geometry_units='meters')
            self.assertTrue(np.all(np.isfinite(result['density_abs'])))

    def test_factor_fallback_releases_mixed_lu_before_allocating_double(self):
        refs = []
        class Stalled:
            def __init__(self, a):
                self.lu = a.astype(np.complex64)
                refs.append(weakref.ref(self.lu))
            def solve(self, rhs, return_residual=False):
                raise np.linalg.LinAlgError('stalled')
        original = rcs._SCIPY_LINALG.lu_factor
        def factor(*args, **kw):
            self.assertEqual(len(refs), 1)
            self.assertIsNone(refs[0]())
            return original(*args, **kw)
        from ghost_backend.linalg.refined_lu import linear_precision
        with linear_precision('mixed'), mock.patch.object(rcs, 'RefinedLU', Stalled), \
             mock.patch.object(rcs._SCIPY_LINALG, 'lu_factor', factor):
            actual = DenseFactor(np.eye(8, dtype=complex)).solve(np.ones((8, 3)))
        np.testing.assert_array_equal(actual, np.ones((8, 3)))


if __name__ == '__main__':
    unittest.main()
