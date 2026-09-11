"""Numerical and lifecycle qualification of the opt-in CPU method (Python 3.6+)."""
import copy
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(name, '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import ghost_backend.twod.solver as rcs
import ghost_backend.execution.cpu as execution
import ghost_backend.twod.assembly.kernels as kernels
from ghost_backend.linalg.refined_lu import linear_precision
from test_2d_capability_acceptance import _circle


def fixture(kind, count=64):
    s = dict(segments=[_circle('body', .06, count, 2)], ibcs=[], dielectrics=[])
    if kind == 'pec':
        return s
    if kind == 'ibc':
        s['segments'][0] = _circle('body', .06, count, 2, ibc=1)
        s['ibcs'] = [['1', 'constant', '50', '-10', '50', '-10']]
        return s
    s['segments'][0] = _circle('body', .06, count, 3, pos=1)
    s['dielectrics'] = [['1', '3', '-0.1', '1', '0']]
    if kind == 'lossless':
        s['dielectrics'][0][2] = '0'
    if kind == 'magnetic':
        s['dielectrics'][0] = ['1', '12', '-.6', '1.8', '-.03']
    if kind == 'translated':
        s['segments'][0] = _circle('body', .06, count, 3, pos=1, center=(.73, -.41))
    if kind == 'coated':
        s['segments'].append(_circle('core', .0582, count, 4, pos=1, ibc=1))
        s['ibcs'] = [['1', 'constant', '50', '-10', '50', '-10']]
    if kind == 'layered':
        s['segments'].append(_circle('core', .03, count, 5, pos=1, neg=2))
        s['dielectrics'].append(['2', '6', '-.3', '1.3', '-.02'])
    if kind == 'mixed':
        s['segments'].append(_circle('pec', .035, count, 2, center=(.14, .01)))
    return s


def solve(snapshot, method='direct', angles=None, certified=False, **kwargs):
    function = rcs.solve_monostatic_rcs_2d_certified if certified else rcs.solve_monostatic_rcs_2d_survey
    return function(snapshot, [.6], list(np.linspace(0, 180, 519) if angles is None else angles),
                    geometry_units='meters', solver_method=method, **kwargs)


def fields(result, pol):
    return np.array([complex(row['rcs_amp_real'], row['rcs_amp_imag'])
                     for row in result['co_solved_samples'][pol]])


class ExperimentalCPU(unittest.TestCase):
    def assert_fields(self, a, b):
        for pol in ('VV', 'HH'):
            aa, bb = fields(a, pol), fields(b, pol)
            peak = max(float(np.max(abs(bb))), 1e-280)
            self.assertLess(float(np.max(abs(aa-bb))) / peak, 1e-10)
            mask = abs(bb) > peak * 1e-6
            self.assertLess(float(np.max(abs(aa[mask]-bb[mask]) / abs(bb[mask]))), 1e-8)

    def test_material_and_pec_fields_all_angles(self):
        for kind in ('pec', 'ibc', 'lossy', 'lossless', 'magnetic', 'translated', 'coated', 'layered', 'mixed'):
            with self.subTest(kind=kind):
                s = fixture(kind, 96 if kind == 'magnetic' else 64)
                reference = solve(s)
                result = solve(s, 'experimental_cpu')
                self.assert_fields(result, reference)
                self.assertTrue(result['metadata']['condition_est_computed'])
                evidence = result['metadata']['experimental_cpu']
                self.assertEqual(len(evidence['systems']), 2)
                for system in evidence['systems']:
                    self.assertEqual(system['factorizations'], 1)
                    self.assertEqual(system['rhs_batches'], 3)
                    self.assertEqual(system['max_rhs_columns'], 256)
                self.assertLessEqual(evidence['cache']['peak_bytes'], execution.CACHE_BYTES)
                self.assertIsNone(execution.current_state())

    def test_certification_and_frequency_dispatch(self):
        s = fixture('lossy', 48)
        args = dict(geometry_units='meters')
        reference = rcs.solve_monostatic_rcs_2d_certified(s, [.6, .8], [0., 17.3, 90., 180.], **args)
        result = rcs.solve_monostatic_rcs_2d_certified(s, [.6, .8], [0., 17.3, 90., 180.], solver_method='experimental_cpu', **args)
        self.assert_fields(result, reference)
        self.assertTrue(result['metadata']['mesh_convergence_certified'])
        self.assertEqual(result['metadata']['solver_method_requested'], 'experimental_cpu')
        self.assertEqual(len(result['metadata']['experimental_cpu']['systems']), 8)

    def test_cancellation_between_batches_restores_state(self):
        event = threading.Event()
        messages = []
        def progress(done, total, message):
            if 'Experimental CPU: solved' in message:
                messages.append(message)
                event.set()
        with self.assertRaises(InterruptedError):
            solve(fixture('pec', 24), 'experimental_cpu', abort_event=event, progress_callback=progress)
        self.assertEqual(len(messages), 1)
        self.assertIsNone(execution.current_state())
        result = solve(fixture('pec', 24), angles=[0.])
        self.assertNotIn('experimental_cpu', result['metadata'])

    def test_concurrent_reference_and_experimental_are_isolated(self):
        s = fixture('lossy', 24)
        with ThreadPoolExecutor(max_workers=2) as pool:
            ref = pool.submit(solve, s, 'direct', [0., 73., 180.])
            exp = pool.submit(solve, copy.deepcopy(s), 'experimental_cpu', [0., 73., 180.])
            a, b = exp.result(), ref.result()
        self.assert_fields(a, b)
        self.assertNotIn('experimental_cpu', b['metadata'])
        self.assertIsNone(execution.current_state())

    def test_cpu_only_and_precision_conflict(self):
        with mock.patch.dict(os.environ, {'GHOST_DENSE_BACKEND': 'gpu'}):
            with mock.patch.object(rcs, '_probe_cupy_backend', side_effect=AssertionError('GPU requested')):
                solve(fixture('pec', 24), 'experimental_cpu', [0., 180.])
        with linear_precision('mixed'):
            with self.assertRaisesRegex(ValueError, 'double'):
                solve(fixture('pec'), 'experimental_cpu', [0.])

    def test_memory_and_quality_gates_still_reject(self):
        with mock.patch.object(rcs, '_solve_memory_limit_gb', return_value=1e-12):
            with self.assertRaises(MemoryError):
                solve(fixture('pec', 24), 'experimental_cpu', [0.])
        with mock.patch.object(rcs, '_equilibrated_condition_from_lu', return_value=1e30):
            with self.assertRaises((ValueError, RuntimeError)):
                solve(fixture('pec', 24), 'experimental_cpu', [0.])
        self.assertIsNone(execution.current_state())

    def test_table_accuracy_and_forced_rejection(self):
        table = kernels.KernelTable(80-2j, .3)
        rr = np.exp(np.random.RandomState(921).uniform(np.log(5e-13), np.log(.3), 10001))
        ref = kernels.values(80-2j, rr)
        got = np.stack([p(rr) for p in table.polys], axis=-1)
        self.assertLess(float(np.max(abs(got-ref) / abs(ref))), 2e-13)
        s = fixture('lossy', 24)
        reference = solve(s, angles=[0., 90., 180.])
        with mock.patch.object(kernels, 'KernelTable', side_effect=kernels.Rejected('qualification rejection')):
            result = solve(s, 'experimental_cpu', [0., 90., 180.])
        self.assert_fields(result, reference)
        self.assertTrue(result['metadata']['experimental_cpu']['kernel_tables'])
        self.assertTrue(all(not t['used'] for t in result['metadata']['experimental_cpu']['kernel_tables']))

    def test_memory_estimate_accounts_for_common_batches(self):
        args = dict(nnodes=512, use_cfie=False, system_dofs=1024, operator_matrices=5, n_rhs=18001)
        normal = rcs._estimate_memory_gb(**args)
        experimental = rcs._estimate_memory_gb(solver_method='experimental_cpu', formulation='single_dielectric', **args)
        fallback = rcs._estimate_memory_gb(solver_method='experimental_cpu', formulation='sheet', **args)
        self.assertGreater(experimental, 2*1024**2*16/1024**3)
        self.assertGreater(fallback, 2*1024**2*16/1024**3)
        short = rcs._estimate_memory_gb(solver_method='experimental_cpu', formulation='single_dielectric',
                                      **dict(args, n_rhs=256))
        self.assertAlmostEqual(experimental-short, (18001-256)*4096/1024**3)

    def test_sheet_uses_common_streamed_solver(self):
        from ghost_backend.geometry.io import parse_geometry, build_geometry_snapshot
        path = Path(__file__).resolve().parents[1]/'validation/thin_dielectric_sheet/thin_strip.geo'
        snapshot = build_geometry_snapshot(*parse_geometry(path.read_text()))
        reference = solve(snapshot, angles=[0., 90., 180.], material_base_dir=str(path.parent))
        result = solve(snapshot, 'experimental_cpu', [0., 90., 180.], material_base_dir=str(path.parent))
        self.assert_fields(result, reference)
        evidence = result['metadata']['experimental_cpu']
        self.assertEqual(len(evidence['systems']), 2)
        self.assertTrue(all(f['streamed'] for f in evidence['formulations']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
