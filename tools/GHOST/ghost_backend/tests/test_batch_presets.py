"""Batch presets, fast planning, and throughput choices preserve solver contracts."""
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND.parent))
from ghost_backend.execution.options import execution_scope, validate_options, current_options
from ghost_backend.runs.config import configuration_payload, driver_contract
from ghost_backend.runs.presets import resolve_preset
from ghost_backend.runs.batch import select_batch_backends, combine_channels
from ghost_backend.hpc import scheduler


def unit(name, dense_ram, compressed_ram, cost=10.):
    return dict(unit=name, backend_candidates=dict(
        dense=dict(peak_gb=dense_ram, cost=cost),
        compressed=dict(peak_gb=compressed_ram, cost=1.4*cost)))


class BatchPresetTests(unittest.TestCase):
    def test_backend_selection_without_math_prod(self):
        # Emulate the legacy HPC math module without changing other modules.
        legacy_math = SimpleNamespace(isfinite=math.isfinite)
        options = validate_options(dict(assembly_threads=1, blas_threads=1))
        cases = [
            ([unit('small', 2, 1)], 'dense'),
            ([unit(str(i), 7, 2) for i in range(4)], 'compressed'),
            ([dict(unit='fmm', backend_candidates=dict(
                dense=dict(peak_gb=2., cost=10.),
                compressed=dict(peak_gb=1., cost=14.),
                fmm=dict(peak_gb=1., cost=2.)))], 'fmm'),
        ]
        with mock.patch('ghost_backend.runs.batch.math', legacy_math), \
                mock.patch('ghost_backend.execution.policy.native_fmm_available', return_value=True):
            for records, expected in cases:
                with self.subTest(expected=expected):
                    choices, summary = select_batch_backends(records, 4, 4, 8, options)
                    self.assertTrue(all(c['selected'] == expected for c in choices.values()))
                    self.assertEqual(summary['search'], 'all_backend_combinations')

    def test_backend_search_cap_and_fixed_choices(self):
        options = validate_options(dict(assembly_threads=1, blas_threads=1))
        for count, search in [(0, 'all_backend_combinations'),
                              (12, 'all_backend_combinations'),
                              (13, 'bounded_mixed_schedules')]:
            with self.subTest(flexible_units=count):
                records = [unit(str(i), 2, 1) for i in range(count)]
                records.append(dict(unit='fixed', backend_candidates=dict(
                    dense=dict(peak_gb=2., cost=10.))))
                choices, summary = select_batch_backends(records, 1, 1, 8, options)
                self.assertEqual(summary['search'], search)
                self.assertEqual(len(choices), count + 1)
                self.assertTrue(all(c['selected'] == 'dense' for c in choices.values()))
                self.assertEqual(summary['selected_cost'], 10. * (count + 1))

    def test_presets_resolve_in_both_drivers_and_survive_json_roundtrip(self):
        for driver in ('run_hpc_monostatic.py', 'run_local_monostatic.py'):
            kind, keys = driver_contract(BACKEND / driver)
            for name, factor, method in [('auto', 'adaptive', 'auto'),
                    ('small', 'dense', 'direct'), ('balanced', 'dense', 'experimental_cpu'),
                    ('large', 'compressed', 'experimental_cpu')]:
                raw = dict(SOLVE_PRESET=name, ADVANCED_OVERRIDES={'assembly_threads': 2},
                           MAX_SOLVE_GB=24, MESH_CERTIFICATION=True, ACCURACY_TARGET='tight')
                payload = configuration_payload(kind, raw, keys)
                settings = payload['settings']
                self.assertEqual(settings['EXECUTION_OPTIONS']['factorization'], factor)
                self.assertEqual(settings['SOLVER_METHOD'], method)
                self.assertEqual(settings['EXECUTION_OPTIONS']['ram_budget_gib'], 24)
                self.assertEqual(settings['ASSEMBLY_THREADS'], 2)
                self.assertTrue(settings['MESH_CERTIFICATION'])
                self.assertEqual(settings['ACCURACY_TARGET'], 'tight')
                self.assertEqual(configuration_payload(kind, json.loads(json.dumps(settings)), keys), payload)

    def test_legacy_recipes_remain_explicit_and_conflicts_are_rejected(self):
        kind, keys = driver_contract(BACKEND / 'run_hpc_monostatic.py')
        legacy = configuration_payload(kind, dict(LU_PRECISION='mixed'), keys)['settings']
        self.assertEqual(legacy['SOLVE_PRESET'], 'custom')
        self.assertEqual(legacy['SOLVER_METHOD'], 'direct')
        for raw in (dict(SOLVE_PRESET='typo'), dict(SOLVE_PRESET='small', LU_PRECISION='mixed'),
                    dict(SOLVE_PRESET='auto', ADVANCED_OVERRIDES={'typo': 1}),
                    dict(SOLVE_PRESET='auto', MAX_SOLVE_GB=4, ADVANCED_OVERRIDES={'ram_budget_gib': 8})):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                configuration_payload(kind, raw, keys)

    def test_portable_request_preserves_presets(self):
        from ghost_backend.hpc.bundle import _validate_settings
        settings = _validate_settings('2d', dict(SOLVE_PRESET='auto',
            ADVANCED_OVERRIDES={'blas_threads': 1, 'angle_batch_size': 32}))
        self.assertEqual(settings['EXECUTION_OPTIONS']['factorization'], 'adaptive')
        self.assertEqual(settings['BLAS_THREADS_PER_WORKER'], 1)
        self.assertEqual(_validate_settings('2d', settings), settings)

    def test_auto_prefers_fast_dense_when_parallelism_cannot_improve(self):
        options = validate_options(dict(assembly_threads=1, blas_threads=1))
        for workers in (1, 4):
            records = [unit('a', 2, 1)] if workers == 1 else [unit(str(i), 2, 1) for i in range(4)]
            choices, summary = select_batch_backends(records, 4, workers, 32, options)
            self.assertTrue(all(c['selected'] == 'dense' for c in choices.values()))
            self.assertEqual(summary['selected_cost'], summary['dense_first_cost'])

    def test_auto_trades_per_solve_speed_for_faster_whole_batch(self):
        options = validate_options(dict(assembly_threads=1, blas_threads=1))
        records = [unit(str(i), 7, 2) for i in range(4)]
        choices, summary = select_batch_backends(records, 4, 4, 8, options)
        self.assertEqual(summary['compressed_units'], 4)
        self.assertLess(summary['selected_cost'], summary['dense_first_cost'])
        self.assertEqual(summary['selected_cost'], 14.)
        # Restricting CPUs prevents the apparent compressed concurrency gain.
        _, serial = select_batch_backends(records, 1, 4, 8, options)
        self.assertEqual(serial['dense_units'], 4)

    def test_mixed_schedule_and_explicit_limits(self):
        options = validate_options(dict(assembly_threads=1, blas_threads=1))
        records = [unit('heavy', 7, 2, 10), unit('light', 2, 1, 10)]
        choices, summary = select_batch_backends(records, 2, 2, 8, options)
        self.assertLess(summary['selected_cost'], summary['dense_first_cost'])
        self.assertEqual(choices['heavy']['selected'], 'compressed')
        options['ram_budget_gib'] = 3
        choices, _ = select_batch_backends(records, 2, 2, 64, options)
        self.assertEqual(choices['heavy']['selected'], 'compressed')
        # Existing oversized-unit fail-loud behavior must not deadlock planning.
        choices, _ = select_batch_backends([unit('oversized', 50, 30)], 4, 4, 8, options)
        self.assertEqual(choices['oversized']['selected'], 'compressed')

    def test_batch_plan_has_no_tile_evaluation_and_reuses_meshes(self):
        from ghost_backend.twod import solver
        geometry = ('Title: planner\nSegment: square 2\nproperties: 2 32 0 0 0\n'
                    '-.02 -.02 -.02 .02\n-.02 .02 .02 .02\n.02 .02 .02 -.02\n'
                    '.02 -.02 -.02 -.02\nIBCS_Resistances:\nDielectrics:\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.geo'
            path.write_text(geometry)
            with execution_scope(dict(factorization='adaptive')), \
                    mock.patch('ghost_backend.compressed.memory.geometry_storage', side_effect=AssertionError('sampled')), \
                    mock.patch.object(solver, '_build_linear_mesh_interface_aware', wraps=solver._build_linear_mesh_interface_aware) as mesh:
                plans = scheduler.predict_2d_resources_many(str(path), [.6, .8], ['TM', 'TE'],
                    'meters', 50000, fine_factor=1.5, n_angles=19, solver_method='experimental_cpu')
            self.assertEqual(mesh.call_count, 4)  # one base/fine topology per frequency
            self.assertEqual(len(plans), 4)
            for p in plans.values():
                self.assertEqual(set(p['backend_candidates']), {'dense', 'compressed','fmm'})
                self.assertFalse(p['backend_candidates']['compressed']['memory_estimate']['sampled'])
                self.assertGreater(p['peak_gb'], 0)

    def test_node_choice_reaches_solver_without_replanning_or_changing_fields(self):
        import numpy as np
        from ghost_backend.twod import solver
        from ghost_backend.execution.selection import batch_selection_scope, current_batch_selection
        from ghost_backend.execution.provenance import runtime_environment_payload
        from test_experimental_cpu import fixture, fields
        args = dict(geometry_snapshot=fixture('pec', 24), frequencies_ghz=[.6],
                    elevations_deg=[0., 90.], geometry_units='meters', solver_method='experimental_cpu')
        reference = solver.solve_monostatic_rcs_2d_survey(**args, execution_options={})
        for mode in ('dense', 'compressed'):
            choice = dict(requested='adaptive', selected=mode, objective='predicted_batch_completion')
            with execution_scope(dict(factorization='adaptive', compressed_storage_mib=64)), \
                    batch_selection_scope(choice), \
                    mock.patch('ghost_backend.execution.selection.select_backend', side_effect=AssertionError('replanned')):
                self.assertEqual(runtime_environment_payload()['cpu_factorization'], 'adaptive')
                result = solver.solve_monostatic_rcs_2d_survey(**args)
                self.assertEqual(result['metadata']['backend_selection'], choice)
                for pol in ('VV', 'HH'):
                    np.testing.assert_allclose(fields(result, pol), fields(reference, pol), rtol=1e-8, atol=1e-12)
            self.assertIsNone(current_options())
            self.assertIsNone(current_batch_selection())


if __name__ == '__main__':
    unittest.main(verbosity=2)
