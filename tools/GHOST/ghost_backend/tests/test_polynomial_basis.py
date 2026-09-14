"""Independent polynomial moments, restricted-space and backend checks."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import hankel2
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_backend.twod import solver as s
from ghost_backend.twod.basis import values, mass_block, derivative_matrix
from ghost_backend.execution.options import execution_scope, validate_options
from ghost_backend.tests.general_fixtures import fixture


def mesh_for(snapshot, degree, freq=1.):
    from ghost_backend.twod.preparation import prepare_geometry
    with execution_scope(validate_options(dict(basis_order=degree))):
        _, _, materials, scale = prepare_geometry(snapshot, None, 'meters')
        wave, _, _ = s._mesh_wavelength_for_snapshot(snapshot, materials, freq)
        panels = s._build_panels(snapshot, scale, wave)
        infos = s._build_coupled_panel_info(panels, materials, freq, 'TE', 2*np.pi*freq*1e9/s.C0)
        return s._build_linear_mesh_interface_aware(panels, infos)[0]


class PolynomialBasisTests(unittest.TestCase):
    def test_polynomial_reproduction_and_mass(self):
        from ghost_backend.twod.basis import abscissae
        for degree in (1, 2, 3):
            x = np.linspace(0, 1, 71)
            phi = values(x, degree)
            for power in range(degree + 1):
                np.testing.assert_allclose(phi @ abscissae(degree)**power, x**power, atol=2e-14)
            mesh = mesh_for(fixture('rectangle', 16), degree)
            element = mesh.elements[0]
            mass = mass_block(element)
            self.assertGreater(np.linalg.eigvalsh(mass).min(), 0.)
            self.assertAlmostEqual(mass.sum(), element.length)
            if degree == 2:
                np.testing.assert_allclose(mass/element.length,
                    np.array([[4,-1,2],[-1,4,2],[2,2,16]])/30, atol=1e-15)

    def test_self_constant_integral_independent_one_dimension(self):
        from ghost_backend.twod.polynomial_quadrature import near_block
        for degree in (2, 3):
            mesh = mesh_for(fixture('rectangle', 16), degree)
            element = mesh.elements[0]
            k = 51 - 7j
            exact = quad_vec(lambda t: 2*(1-t)*.25j*hankel2(0,k*element.length*t),
                             0, 1, epsabs=1e-12, epsrel=1e-12)[0]*element.length**2
            actual = near_block(element, element, k)[0].sum()
            np.testing.assert_allclose(actual, exact, rtol=3e-10, atol=1e-14)

    def test_touching_quadrature_translation_and_scaling(self):
        import copy
        from ghost_backend.twod.polynomial_quadrature import near_block, block
        for degree in (2, 3):
            mesh = mesh_for(fixture('rectangle', 4), degree, freq=.1)
            a, b = mesh.elements[:2]
            expected = block(a, b, 42.-2j, order=144)
            translated = copy.deepcopy((a, b))
            for element in translated:
                element.p0 += np.array([-.29845, .021])
                element.p1 += np.array([-.29845, .021])
            for actual, reference in zip(near_block(*translated, 42.-2j), expected):
                np.testing.assert_allclose(actual, reference, rtol=2e-9, atol=1e-13)
            scaled = copy.deepcopy((a, b))
            for element in scaled:
                element.p0 *= 1e-8
                element.p1 *= 1e-8
                element.length *= 1e-8
            actual = near_block(*scaled, (42.-2j)*1e8)
            np.testing.assert_allclose(actual[0]/1e-16, expected[0], rtol=2e-9, atol=1e-13)
            np.testing.assert_allclose(actual[1]/1e-8, expected[1], rtol=2e-9, atol=1e-13)

    def test_fmm_matches_dense_polynomial_operators(self):
        from ghost_backend.twod.fmm.galerkin import GalerkinKernel
        rng = np.random.default_rng(44)
        for degree in (2, 3):
            mesh = mesh_for(fixture('rectangle', 24), degree)
            kernel = GalerkinKernel(mesh, 42.-2j)
            single, kp = s._assemble_linear_operator_matrices(mesh, 42.-2j, True)
            w = s._assemble_linear_hypersingular_matrix(mesh, 42.-2j)
            x = rng.normal(size=(len(mesh.nodes), 2)) + 1j*rng.normal(size=(len(mesh.nodes), 2))
            for kind, matrix in (('S', single), ('KP', kp), ('K', kp.T), ('W', w)):
                np.testing.assert_allclose(kernel.apply(kind, x), matrix@x, rtol=2e-7, atol=3e-10)

    def test_dense_compressed_fmm_material_and_sheet_solutions(self):
        for degree in (2, 3):
            for case in ('dielectric', 'mixed', 'sheet'):
                results = []
                for mode in ('dense', 'compressed', 'fmm'):
                    result = s.solve_monostatic_rcs_2d(fixture(case, 24), [1.], [0., 47., 93.],
                        geometry_units='meters', compute_condition_number=True,
                        execution_options=dict(basis_order=degree, factorization=mode))
                    self.assertTrue(result['metadata']['quality_gate']['passed'])
                    fields = [complex(row['rcs_amp_real'], row['rcs_amp_imag'])
                              for rows in result['co_solved_samples'].values() for row in rows]
                    results.append(np.asarray(fields))
                for actual in results[1:]:
                    error = np.max(abs(actual-results[0]))/np.max(abs(results[0]))
                    self.assertLess(error, 2e-7, (degree, case))


if __name__ == '__main__': unittest.main()
