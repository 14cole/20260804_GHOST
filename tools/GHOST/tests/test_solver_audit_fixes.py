"""Independent physical anchors and allocation regressions from the 2026 audit."""
from pathlib import Path
import json
import sys
import time
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from scipy import integrate, special

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Backend'))
import rcs_solver as rcs
import bor_solver as bor
import bor_kernels as kernels
import bor_dispatch
from mie_reference import pec_cylinder_backscatter_amplitude
from mie_sphere import sigma_coated_pec_sphere
from test_rcs_physics_regression import _circle_segment


def cylinder_amplitude(pol, eps=3-.1j, core=None):
    """Solve cylindrical field/flux matching, independent of the BIE."""
    radius = .07
    k = 2 * np.pi * 1e9 / rcs.C0
    ki = k * np.sqrt(eps + 0j)
    flux = eps if pol == 'TE' else 1.
    total = 0j
    for n in range(-30, 31):
        jo, jop = special.jv(n, k*radius), special.jvp(n, k*radius)
        ho, hop = special.hankel2(n, k*radius), special.h2vp(n, k*radius)
        ji, jip = special.jv(n, ki*radius), special.jvp(n, ki*radius)
        if core is None:
            a = [[ho, -ji], [k*hop, -ki/flux*jip]]
            rhs = [-jo, -k*jop]
        else:
            yi, yip = special.yv(n, ki*radius), special.yvp(n, ki*radius)
            bcj = special.jv(n, ki*core) if pol == 'TM' else special.jvp(n, ki*core)
            bcy = special.yv(n, ki*core) if pol == 'TM' else special.yvp(n, ki*core)
            a = [[ho, -ji, -yi], [k*hop, -ki/flux*jip, -ki/flux*yip], [0, bcj, bcy]]
            rhs = [-jo, -k*jop, 0]
        total += (-1.)**n * np.linalg.solve(np.asarray(a), rhs)[0]
    return -4j * total


def snapshot(kind='dielectric', count=96, loss=-.1):
    if kind == 'sheet':
        return dict(segments=[_circle_segment(.07, count, 1, ibc=1)],
                    ibcs=[['1', 'constant', '0.000001', '0', '0.000001', '0']], dielectrics=[])
    segments = [_circle_segment(.07, count, 3, material=1)]
    if kind == 'coated':
        segments.append(_circle_segment(.04, count, 4, material=1))
    return dict(segments=segments, ibcs=[], dielectrics=[['1','3',str(loss),'1','0']])


class PhaseAndSweepTests(unittest.TestCase):
    def test_certified_absolute_phase_for_all_affected_formulations(self):
        for kind in ('dielectric', 'coated', 'sheet'):
            for pol in ('TE', 'TM'):
                with self.subTest(kind=kind, pol=pol):
                    result = rcs.solve_monostatic_rcs_2d_certified_single_polarization(
                        snapshot(kind), [1.], [0.], pol, geometry_units='meters')
                    row = result['samples'][0]
                    got = complex(row['rcs_amp_real'], row['rcs_amp_imag'])
                    ref = (pec_cylinder_backscatter_amplitude(.07, 1e9, pol)
                           if kind == 'sheet' else cylinder_amplitude(pol, core=.04 if kind == 'coated' else None))
                    self.assertLess(abs(got/ref-1), .004)
                    self.assertTrue(result['metadata']['mesh_convergence_certified'])
                    self.assertEqual(result['amplitude_version'], 2)

    def test_bistatic_absolute_phase_and_lossless_energy(self):
        for pol in ('TE', 'TM'):
            result = rcs.solve_bistatic_rcs_2d_single_polarization(
                snapshot(loss=0), [1.], [0.], list(np.arange(0.,360.,2.)),
                pol, geometry_units='meters')
            rows = result['samples']
            got = complex(rows[0]['rcs_amp_real'], rows[0]['rcs_amp_imag'])
            self.assertLess(abs(got/cylinder_amplitude(pol, eps=3)-1), .006)
            scattering = np.mean([row['rcs_linear'] for row in rows])
            forward = next(row for row in rows if row['theta_scat_deg'] == 180.)
            extinction = forward['rcs_amp_imag'] / (2*np.pi*1e9/rcs.C0)
            self.assertAlmostEqual(extinction/scattering, 1., delta=2e-5)

    def test_bistatic_gate_accounts_for_every_incident_field(self):
        estimate = rcs._estimate_memory_gb
        counts = []
        def record(*args, **kw):
            counts.append(kw['n_rhs'])
            return estimate(*args, **kw)
        with patch.object(rcs, '_estimate_memory_gb', record):
            rcs.solve_bistatic_rcs_2d_single_polarization(
                snapshot(count=24), [1.], [0.,20.,40.,60.], [0.], 'TM',
                geometry_units='meters', strict_quality_gate=False)
        self.assertEqual(counts, [4])

    def test_frequency_local_caches_reuse_ibc_and_release_pec(self):
        for ibc in (False, True):
            geometry = dict(segments=[_circle_segment(.03,24,2,ibc=int(ibc))],
                            ibcs=[['1','constant','50','10','50','10']] if ibc else [],dielectrics=[])
            calls = []
            original = rcs.solve_monostatic_rcs_2d_single_polarization
            def record(**kw):
                result = original(**kw)
                cache = kw['_shared_discretization_cache']['operators']
                arrays = sum(isinstance(v,np.ndarray) for v in cache.values())
                calls.append((kw['frequencies_ghz'],kw['polarization'],arrays))
                return result
            with patch.object(rcs,'solve_monostatic_rcs_2d_single_polarization',record):
                result = rcs.solve_monostatic_rcs_2d(geometry,[1.,1.1,1.2],[0.,45.],
                    geometry_units='meters',strict_quality_gate=False)
            self.assertEqual([c[1] for c in calls], ['TE','TM']*3)
            self.assertTrue(all(len(c[0]) == 1 for c in calls))
            self.assertLessEqual(max(c[2] for c in calls), int(ibc))
            self.assertEqual(len(result['samples']),12)
            if ibc:
                self.assertGreaterEqual(result['metadata']['shared_operator_cache_hits'],3)

    def test_mesh_density_does_not_depend_on_primitive_segmentation(self):
        for count in (lambda length: rcs._panel_count_from_n(-20,length,.1),
                      lambda length: bor_dispatch._element_count(-20,length,.1),
                      lambda length: bor_dispatch._element_count(0,length,.1)):
            self.assertEqual(count(100.), 20000)
            self.assertEqual(count(100.),10*count(10.))


class BorNearTests(unittest.TestCase):
    def test_vector_near_kernels_match_independent_angular_integration(self):
        rp, zp, rq, zq = 1., 0., 1., .01
        trp, tzp, trq, tzq = .6, .8, .8, -.6
        for k, mode in ((3-.2j,5), (300-2j,312)):
            args = (rp,zp,trp,tzp,rq,zq,trq,tzq)
            for near, brackets in ((kernels.mfie_kernels_near,kernels._mfie_brackets),
                                   (kernels.ibc_kernels_near,kernels._ibc_brackets_grid)):
                got = near(*args,k,mode)
                for component in range(4):
                    def value_at(x):
                        if brackets is kernels._ibc_brackets_grid:
                            value = brackets(*(np.array([a]) for a in args),k,np.array([[x]]))[component][0,0]
                        else:
                            value = brackets(*(np.asarray(a) for a in args),k,np.asarray(x))[component]
                        return complex(np.asarray(value).item())*np.exp(-1j*mode*x)
                    # Combine the +/- halves before integration to avoid
                    # cancellation of large odd terms across the whole cut.
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', integrate.IntegrationWarning)
                        reference = integrate.quad(lambda x: value_at(x)+value_at(-x),
                            0,np.pi,points=[.01,.1],complex_func=True,
                            epsabs=1e-9,epsrel=1e-9,limit=4000)[0]
                    self.assertLess(abs(got[component][0,2*mode]-reference),1e-7*max(1.,abs(reference)))

    def test_vector_axis_pairs_use_regular_angular_rule(self):
        args = (np.array([.003]),np.array([.01]),1.,0.,np.array([0.]),np.array([0.]),1.,0.)
        for near in (kernels.mfie_kernels_near,kernels.ibc_kernels_near):
            value = near(*args,3.,6)
            self.assertTrue(all(np.all(np.isfinite(v)) for v in value))

    def test_fft_point_tiles_do_not_change_vector_kernels(self):
        rho = np.linspace(.1,.8,32)
        z = np.linspace(-.2,.3,32)
        args = (rho[:,None],z[:,None],.6,.8,rho[None,:]+.02,z[None,:]+.07,.8,-.6)
        expected = kernels.mfie_kernels_fft(*args,30.,8,n_xi=128)
        with patch.object(kernels,'FFT_BUILD_BUDGET',1_000_000):
            actual = kernels.mfie_kernels_fft(*args,30.,8,n_xi=128)
        for a,b in zip(actual,expected):
            np.testing.assert_allclose(a,b,rtol=1e-13,atol=1e-13)

    def test_high_mode_kernels_match_independent_adaptive_integral(self):
        for k, mode in ((300.,312),(500.,256),(500.,512)):
            gap = .01
            def integrand(x):
                distance = np.sqrt(gap*gap + 4*np.sin(x/2)**2)
                return 2*np.exp(-1j*k*distance)*np.cos(mode*x)/(4*np.pi*distance)
            reference = integrate.quad(integrand,0,np.pi,complex_func=True,
                                       epsabs=2e-11,epsrel=2e-11,limit=4000)[0]
            got = kernels.modal_kernels_near([1.],[0.],[1.],[gap],k,mode)[0,mode]
            self.assertLess(abs(got/reference-1),1e-8)

    def test_angular_order_limit_rejects_inaccurate_result(self):
        with patch.object(kernels,'NEAR_ANGULAR_MAX_ORDER',128):
            with self.assertRaisesRegex(ValueError,'accuracy limit'):
                kernels.modal_kernels_near([1.],[0.],[1.],[.01],500.,512)

    def test_preparation_releases_raw_kernels_and_preserves_operators(self):
        solver = bor.BorPecSolver(bor.sphere_generatrix(.03,8),1e9)
        expected = solver.assemble_mode(1,2)
        solver.prepare_operators(2,efie=True,mfie=True,ibc=True)
        self.assertFalse(solver._near_cache)
        np.testing.assert_allclose(solver.assemble_mode(1,2),expected,rtol=2e-9,atol=1e-12)
        retained = sum(v['values'].nbytes for v in solver._near_contractions.values())
        budget = bor.estimate_bor_operator_storage_gb(2,((solver,True,True,True),),streaming=True)*1e9
        self.assertLess(retained,budget)

    def test_sparse_constraint_reduction_matches_dense_algebra(self):
        rng = np.random.default_rng(99)
        a = rng.normal(size=(30,30)) + 1j*rng.normal(size=(30,30))
        q = np.eye(30,dtype=complex)[:,:27]
        q[28,0] = 1j
        q[29,26] = -1j
        np.testing.assert_allclose(bor._reduce_constrained_operator(a,q), q.conj().T@a@q,
                                   rtol=1e-14,atol=1e-14)

    def test_thin_sphere_agrees_with_analytic_coating(self):
        started = time.perf_counter()
        result = bor.solve_bor_coated_pec(bor.sphere_generatrix(.07,24),
            bor.sphere_generatrix(.0699,24),1e9,[0.,60.,90.],3-.1j,1.,table_precision='double')
        reference = sigma_coated_pec_sphere(.0699,.07,3-.1j,1.,1e9)
        values = np.array([result['sigma_vv'],result['sigma_hh']])
        error_db = float(np.max(np.abs(10*np.log10(values/reference))))
        self.assertLess(error_db,.08)
        self.assertLessEqual(result['near_quadrature']['disjoint_meridian_relative_change_max'],2e-5)
        print('THIN_COATING_AUDIT ' + json.dumps(dict(
            max_error_db=error_db, sigma=values.tolist(), reference=reference,
            near_quadrature=result['near_quadrature'], seconds=time.perf_counter()-started)),flush=True)


if __name__ == '__main__':
    unittest.main()
