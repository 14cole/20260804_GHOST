"""New-run defaults and explicit saved-profile compatibility."""
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.execution.options import DEFAULTS, efficient_defaults, from_environment, validate_options, geometry_preset
from ghost_backend.runs.execution import driver_options
from ghost_backend.runs.config import configuration_payload


class EfficientDefaultsTests(unittest.TestCase):
    def test_geometry_presets_preserve_certified_pec_and_mixed_fields(self):
        import numpy as np
        from ghost_backend.twod.solver import solve_monostatic_rcs_2d_certified
        from test_experimental_cpu import fixture, fields
        for material in ('pec', 'mixed'):
            results = []
            for name in ('small', 'balanced', 'large'):
                preset = geometry_preset(name)
                result = solve_monostatic_rcs_2d_certified(fixture(material,64), [.6], list(range(361)),
                    geometry_units='meters', solver_method=preset['solver_method'], execution_options=preset['execution_options'])
                self.assertTrue(result['metadata']['mesh_convergence_certified'])
                results.append(result)
            for result in results[1:]:
                for pol in ('VV','HH'):
                    a,b = fields(results[0],pol),fields(result,pol)
                    self.assertLess(float(np.max(abs(a-b))/np.max(abs(a))), 1e-10)
            self.assertEqual(results[-1]['metadata']['execution_threads'],
                dict(assembly=efficient_defaults()['assembly_threads'],blas=efficient_defaults()['blas_threads']))

    def test_preset_and_launch_overrides_do_not_change_serialized_v1_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(os, 'cpu_count', return_value=8):
            profile = efficient_defaults()
            self.assertEqual((profile['factorization'], profile['compressed_storage_mib'],
                              profile['assembly_threads'], profile['blas_threads']), ('compressed',8192,4,2))
            self.assertEqual((profile['rhs_compression'], profile['angle_batch_size']), ('auto',256))
            self.assertIsNone(profile['ram_budget_gib'])
            self.assertEqual(from_environment(profile), profile)
            self.assertEqual(validate_options({}), DEFAULTS)
            self.assertEqual(from_environment()['factorization'], 'dense')
            with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION':'dense', 'OPENBLAS_NUM_THREADS':'1'}):
                changed = from_environment(profile)
                self.assertEqual((changed['factorization'],changed['blas_threads']), ('dense',1))
            profile['assembly_threads'] = 99
            self.assertEqual(efficient_defaults()['assembly_threads'], 4)
        with mock.patch.object(os, 'cpu_count', return_value=1):
            self.assertEqual(efficient_defaults()['blas_threads'], 1)
            self.assertEqual(efficient_defaults()['assembly_threads'], 1)

    def test_new_drivers_use_preset_and_explicit_reference_profiles_still_work(self):
        import run_local_monostatic as local
        import run_hpc_monostatic as hpc
        with mock.patch.dict(os.environ, {}, clear=True):
            for driver in (local,hpc):
                self.assertEqual(driver.SOLVER_METHOD, 'experimental_cpu')
                self.assertEqual(driver.LU_PRECISION, 'double')
                self.assertTrue(driver.MESH_CERTIFICATION)
                profile = driver_options(vars(driver))
                self.assertEqual(profile, efficient_defaults())
                explicit = dict(vars(driver), EXECUTION_OPTIONS=validate_options(dict(factorization='dense')),
                                SOLVER_METHOD='direct', LU_PRECISION='mixed')
                self.assertEqual(driver_options(explicit)['factorization'], 'dense')
                self.assertEqual(driver_options(dict(vars(driver), SOLVER_METHOD='direct'))['factorization'], 'dense')
            payload = configuration_payload('2d', {'LU_PRECISION':'mixed'}, local._CONFIG_KEYS)
            self.assertEqual(payload['settings']['SOLVER_METHOD'], 'direct')


if __name__ == '__main__':
    unittest.main()
