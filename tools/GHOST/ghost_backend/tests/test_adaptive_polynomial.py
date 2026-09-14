"""Production h/p acceptance, resource planning and explicit-override behavior."""
from pathlib import Path
import copy
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.tests.general_fixtures import fixture
from ghost_backend.twod import solver as s
from ghost_backend.execution.options import validate_options, execution_scope, efficient_defaults
from ghost_backend.execution.selection import select_backend


def wavelength_fixture(case='rectangle'):
    counts = dict(rectangle=4, reentrant=8, acute=4, gap=8, dielectric=4, mixed=8, sheet=2)
    result = fixture(case, counts[case])
    for segment in result['segments']: segment['properties'][1] = '-100'
    return result


class AdaptivePolynomialTests(unittest.TestCase):
    def setUp(self):
        # Exercise the numerical controller cheaply; the production small-model
        # policy is checked separately below and in a real HPC worker.
        small = patch('ghost_backend.twod.adaptive_geometry.MIN_AUTOMATIC_REFERENCE_PANELS', 0)
        small.start()
        self.addCleanup(small.stop)

    def test_small_model_keeps_reference_method_automatically(self):
        with patch('ghost_backend.twod.adaptive_geometry.MIN_AUTOMATIC_REFERENCE_PANELS', 1024):
            result = s.solve_monostatic_rcs_2d_certified(wavelength_fixture(), [1.], [0., 90.],
                geometry_units='meters', execution_options=dict(mesh_strategy='adaptive',factorization='adaptive'))
        self.assertFalse(result['metadata']['adaptive_mesh']['used'])
        self.assertIn('Small reference mesh', result['metadata']['adaptive_mesh']['reason'])
        self.assertEqual(result['metadata']['polynomial_degree'], 1)

    def test_default_and_invalid_degree(self):
        self.assertEqual(efficient_defaults()['mesh_strategy'], 'adaptive')
        for degree in (0, 4, True, 2., '2'):
            with self.assertRaises(ValueError): validate_options(dict(basis_order=degree))

    def test_candidate_reduces_unknowns_and_preserves_input(self):
        snapshot = wavelength_fixture()
        before = copy.deepcopy(snapshot)
        args = dict(geometry_snapshot=snapshot, frequencies_ghz=[1.], elevations_deg=[0., 47., 91.],
                    geometry_units='meters')
        adaptive = s.solve_monostatic_rcs_2d_certified(**args,
            execution_options=dict(mesh_strategy='adaptive', factorization='dense'))
        reference = s.solve_monostatic_rcs_2d_certified(**args,
            execution_options=dict(mesh_strategy='global', factorization='dense'))
        self.assertEqual(snapshot, before)
        self.assertTrue(adaptive['metadata']['mesh_convergence_certified'])
        self.assertTrue(adaptive['metadata']['adaptive_mesh']['used'])
        self.assertEqual(adaptive['metadata']['polynomial_degree'], 3)
        self.assertEqual(adaptive['metadata']['execution_options']['basis_order'], 3)
        self.assertLessEqual(adaptive['metadata']['adaptive_mesh']['acceptance_policy']['complex_max_limit'], .002)
        self.assertLess(adaptive['metadata']['linear_node_count'], reference['metadata']['linear_node_count'])
        for pol in ('VV','HH'):
            def fields(result):
                return np.array([complex(row['rcs_amp_real'],row['rcs_amp_imag']) for row in result['co_solved_samples'][pol]])
            error = np.max(abs(fields(adaptive)-fields(reference)))/np.max(abs(fields(reference)))
            self.assertLess(error, .002)

    def test_planner_has_same_polynomial_mesh_as_execution(self):
        snapshot = wavelength_fixture('dielectric')
        args = dict(geometry_snapshot=snapshot, frequencies_ghz=[1.], elevations_deg=[0., 61.], geometry_units='meters')
        options = validate_options(dict(mesh_strategy='adaptive', factorization='adaptive'))
        plan = select_backend(args, options, certified=True)
        self.assertEqual({record['polynomial_degree'] for record in plan['meshes']}, {2,3})
        result = s.solve_monostatic_rcs_2d_certified(**args, execution_options=options)
        steps = result['metadata']['adaptive_mesh']['steps']
        self.assertTrue(steps)
        self.assertEqual(steps[0]['panels'], plan['meshes'][0]['panels'])
        self.assertTrue(all(step['backend_selection'] for step in steps))

    def test_admitted_whole_request_backend_preference_is_retained(self):
        def prefer_compressed(selection, key, batch=False):
            self.assertIn('compressed', selection['candidates'])
            return dict(selection, selected='compressed', retry_order=['dense'])
        with patch('ghost_backend.execution.timing_history.adjust', prefer_compressed):
            result = s.solve_monostatic_rcs_2d_certified(wavelength_fixture(), [1.], [0.,47.],
                geometry_units='meters', execution_options=dict(mesh_strategy='adaptive',factorization='adaptive'))
        self.assertTrue(result['metadata']['adaptive_mesh']['used'])
        self.assertTrue(all(step['backend']=='compressed' for step in result['metadata']['adaptive_mesh']['steps']))

    def test_frequency_sweep_forecasts_each_side_of_size_crossover(self):
        args = dict(geometry_snapshot=wavelength_fixture(), frequencies_ghz=[.5,1.],
                    elevations_deg=[0.,47.], geometry_units='meters')
        options = validate_options(dict(mesh_strategy='adaptive', factorization='adaptive'))
        with patch('ghost_backend.twod.adaptive_geometry.MIN_AUTOMATIC_REFERENCE_PANELS',100):
            plan = select_backend(args, options, certified=True)
            result = s.solve_monostatic_rcs_2d_certified(**args, execution_options=options)
        self.assertEqual({m['polynomial_degree'] for m in plan['meshes'] if m['frequency_ghz']==.5},{1})
        self.assertEqual({m['polynomial_degree'] for m in plan['meshes'] if m['frequency_ghz']==1.},{2,3})
        self.assertEqual([r['metadata']['polynomial_degree'] for r in result['metadata']['frequency_metadata']],[1,3])
        self.assertEqual(result['metadata']['polynomial_degree_min'],1)
        self.assertEqual(result['metadata']['polynomial_degree_max'],3)

    def test_failed_comparison_refines_and_then_uses_reference(self):
        snapshot = wavelength_fixture()
        real_finish = s._finish_certified_2d_pair
        seen = []
        def finish(base, fine, policy):
            from ghost_backend.execution.options import option
            if option('basis_order',1) > 1:
                seen.append(fine['metadata']['panel_count'])
                raise ValueError('Certified 2-D mesh convergence failed: injected unresolved field error')
            return real_finish(base, fine, policy)
        with patch.object(s, '_finish_certified_2d_pair', finish):
            result = s.solve_monostatic_rcs_2d_certified(snapshot, [1.], [0.,67.], geometry_units='meters',
                execution_options=dict(mesh_strategy='adaptive', factorization='dense'))
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(b > a for a,b in zip(seen,seen[1:])))
        self.assertTrue(result['metadata']['adaptive_mesh']['fallback'])
        self.assertEqual(result['metadata']['polynomial_degree'], 1)
        self.assertTrue(result['metadata']['mesh_convergence_certified'])

    def test_fixed_panel_override_is_preserved(self):
        result = s.solve_monostatic_rcs_2d_certified(fixture('rectangle',24),[1.],[0.,90.], geometry_units='meters',
            execution_options=dict(mesh_strategy='adaptive', factorization='dense'))
        self.assertFalse(result['metadata']['adaptive_mesh']['used'])
        self.assertEqual(result['metadata']['polynomial_degree'],1)


if __name__ == '__main__': unittest.main()
