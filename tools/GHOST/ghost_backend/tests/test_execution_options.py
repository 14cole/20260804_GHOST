"""Saved execution profiles, scoped settings, worker portability, and fields."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND.parent))
from ghost_backend.execution.options import (
    current_options, execution_scope, option, validate_options, validate_for_run, configured_execution,
)
from ghost_backend.execution.metrics import progress_listener
from ghost_backend.linalg.hierarchical import factor_mode
from ghost_backend.linalg.sweep import mode as sweep_mode
from ghost_backend.compressed.runtime import storage_budget
from ghost_backend.runs.execution import reconcile_settings, driver_options
from ghost_backend.runs.setup import DEFAULT_QUALITY, read_setup, save_setup, validate_setup
from ghost_backend.twod import operators, solver
from test_experimental_cpu import fixture, fields


def setup_record(options=None):
    return dict(schema='grim.2d-run-setup', version=2, frequencies_ghz=[.6],
                angles_deg=[0., 90., 360.], units='meters', mesh_certification=True,
                accuracy='standard', lu_precision='double', scattering='monostatic',
                observation_angles_deg=[], quality=dict(DEFAULT_QUALITY),
                solver_method='experimental_cpu', execution_options=validate_options(options or {}))


class ExecutionOptionsTests(unittest.TestCase):
    def test_uncaptured_driver_keeps_environment_memory_override(self):
        with mock.patch.dict(os.environ, {'GHOST_MAX_SOLVE_GB': '12.5'}):
            profile = driver_options(dict(MAX_SOLVE_GB=None, BLAS_THREADS_PER_WORKER=1,
                                          ASSEMBLY_THREADS='auto'))
            self.assertEqual(profile['ram_budget_gib'], 12.5)

    def test_profile_budget_respects_available_memory_and_nested_options(self):
        with execution_scope(dict(ram_budget_gib=48)), mock.patch.object(solver, '_detect_available_gb', return_value=10):
            self.assertEqual(solver._solve_memory_limit_gb(), 9)
            @configured_execution
            def nested():
                return {}
            with self.assertRaisesRegex(ValueError, 'nested solve'):
                nested(execution_options=dict(ram_budget_gib=8))

    def test_scheduler_allocation_limits_assembly_and_auto_is_deterministic(self):
        with mock.patch.object(operators, '_ASSEMBLY_THREADS', 17):
            with execution_scope(dict(assembly_threads='auto')):
                self.assertEqual(operators.get_assembly_threads(), 1)
            for requested in ('auto', 9):
                with execution_scope(dict(assembly_threads=requested), assembly_threads=2):
                    values = []
                    operators._run_tiled_obs_blocks(8, 2,
                        lambda a,b: values.append(operators.get_assembly_threads()))
                    self.assertEqual(values, [2]*4)

    def test_live_metrics_keep_reporting_during_a_blocking_stage(self):
        from ghost_backend.execution.metrics import SolveMetrics
        arrived = threading.Event()
        events = []
        def collect(event):
            events.append(event)
            if event['stage_key'] == 'factorization':
                arrived.set()
        with progress_listener(collect):
            metrics = SolveMetrics()
            metrics.start()
            try:
                with metrics.stage('factorization'):
                    self.assertTrue(arrived.wait(3))
            finally:
                metrics.finish()
        self.assertFalse(metrics._sampler.is_alive())
        self.assertGreater(metrics.report()['stage_seconds']['factorization'], 0)

    def test_saved_profile_overrides_launch_environment_and_restores_scope(self):
        with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION': 'compressed',
                                         'GHOST_COMPRESSED_STORAGE_MIB': '999',
                                         'GHOST_CPU_RHS_COMPRESSION': 'on'}):
            profile = validate_options(dict(factorization='dense', compressed_storage_mib=128,
                                            rhs_compression='off', assembly_threads=3))
            with execution_scope(profile):
                profile['factorization'] = 'bad'
                self.assertEqual(factor_mode(), 'dense')
                self.assertEqual(storage_budget(), 128*1024**2)
                self.assertEqual(sweep_mode(), 'off')
                self.assertEqual(operators.get_assembly_threads(), 3)
                with self.assertRaises(RuntimeError):
                    with execution_scope(dict(factorization='hierarchical')):
                        raise RuntimeError('stop')
                self.assertEqual(factor_mode(), 'dense')
            self.assertIsNone(current_options())
            self.assertEqual(factor_mode(), 'compressed')

    def test_threads_and_tiled_workers_receive_independent_profiles(self):
        barrier = threading.Barrier(2)
        def task(mode):
            with execution_scope(dict(factorization=mode, assembly_threads=2)):
                barrier.wait(timeout=10)
                values = []
                operators._run_tiled_obs_blocks(8, 2, lambda a,b: values.append(factor_mode()))
                return values
        with ThreadPoolExecutor(max_workers=2) as pool:
            for mode, values in zip(('dense', 'compressed'), pool.map(task, ('dense', 'compressed'))):
                self.assertEqual(values, [mode]*4)

    def test_blas_limit_is_applied_and_restored_after_failure(self):
        from threadpoolctl import threadpool_info
        def counts():
            return {row['filepath']: row['num_threads'] for row in threadpool_info() if row['user_api'] == 'blas'}
        before = counts()
        self.assertTrue(before)
        with self.assertRaisesRegex(RuntimeError, 'abort'):
            with execution_scope(dict(blas_threads=2), limit_blas=True):
                self.assertTrue(all(n == 2 for n in counts().values()))
                raise RuntimeError('abort')
        self.assertEqual(counts(), before)

    def test_saved_setup_roundtrip_and_legacy_migration_ignore_environment(self):
        record = setup_record(dict(factorization='compressed', compressed_storage_mib=8192,
                                   ram_budget_gib=8.25, assembly_threads=4, blas_threads=2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'airfoil.run.json'
            save_setup(path, record)
            with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION': 'dense'}):
                self.assertEqual(read_setup(path), record)
                old = dict(record, version=1)
                old.pop('execution_options')
                migrated = validate_setup(old)
                self.assertEqual(migrated['version'], 2)
                self.assertEqual(migrated['execution_options']['factorization'], 'dense')

    def test_invalid_profiles_and_conflicting_driver_values_reject(self):
        for value in ({'factorization': 'typo'}, {'ram_budget_gib': float('nan')},
                      {'blas_threads': True}, {'angle_batch_size': 257},
                      {'temporary_directory': '../temp'}, {'unknown': 1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_options(value)
        for mode, precision, scattering, method in [('compressed','double','monostatic','direct'),
                ('hierarchical','mixed','monostatic','direct'), ('compressed','double','bistatic','experimental_cpu')]:
            with self.assertRaises(ValueError):
                validate_for_run(dict(factorization=mode), method, precision, scattering)
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            reconcile_settings(dict(EXECUTION_OPTIONS=dict(blas_threads=2), BLAS_THREADS_PER_WORKER=4))

    def test_fresh_interpreter_uses_serialized_profile_with_qt_blocked(self):
        script = '''
import json,sys
sys.path.insert(0,sys.argv[1])
class NoGui:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('PySide6','PySide2','PyQt5','PyQt6'):
            raise RuntimeError('Qt imported')
sys.meta_path.insert(0,NoGui())
from ghost_backend.execution.options import execution_scope,current_options
from ghost_backend.linalg.hierarchical import factor_mode
with execution_scope(json.loads(sys.argv[2]),limit_blas=True):
    assert factor_mode()=='compressed'
    print(json.dumps(current_options(),sort_keys=True))
'''
        value = validate_options(dict(factorization='compressed', blas_threads=2))
        child = subprocess.run([sys.executable, '-I', '-c', script, str(BACKEND.parent), json.dumps(value)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30,
            env=dict(os.environ, GHOST_CPU_FACTORIZATION='dense'))
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout), value)

    def test_profiles_preserve_material_fields_and_report_execution(self):
        for kind in ('pec', 'ibc', 'lossy', 'magnetic', 'coated', 'layered', 'mixed'):
            geometry = fixture(kind, 20)
            values = []
            for mode in ('dense', 'hierarchical', 'compressed'):
                with self.subTest(material=kind, mode=mode):
                    profile = validate_options(dict(factorization=mode, assembly_threads=2, blas_threads=1))
                    result = solver.solve_monostatic_rcs_2d_survey(geometry, [.6], [0., 90., 360.],
                        geometry_units='meters', solver_method='experimental_cpu', execution_options=profile)
                    self.assertEqual(result['metadata']['execution_options'], profile)
                    values.append(result)
            for result in values[1:]:
                for pol in ('VV', 'HH'):
                    a, b = fields(result, pol), fields(values[0], pol)
                    self.assertLess(np.max(abs(a-b))/max(np.max(abs(b)), 1e-100), 1e-9)

    def test_certification_reports_phases_and_spools_in_selected_directory(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            profile = dict(factorization='compressed', temporary_directory=directory, blas_threads=1)
            with progress_listener(events.append), mock.patch('ghost_backend.linalg.dense.DenseFactor.__init__', side_effect=AssertionError('dense fallback')):
                result = solver.solve_monostatic_rcs_2d_certified(fixture('pec', 48), [.6], [0., 90.],
                    geometry_units='meters', solver_method='experimental_cpu', execution_options=profile)
            self.assertTrue(result['metadata']['mesh_convergence_certified'])
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertTrue({'Base mesh', 'Refined mesh'} <= {event['phase'] for event in events})
        self.assertTrue(all(event['elapsed_seconds'] >= 0 for event in events))


if __name__ == '__main__':
    unittest.main()
