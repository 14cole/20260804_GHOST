"""Native signs, adjoints, near corrections, material routing and rejection."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import hankel2
sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parent)]
from ghost_backend.twod.fmm.kernel import evaluate,library,NativePlan
from ghost_backend.twod.fmm.galerkin import GalerkinKernel
from ghost_backend.twod.fmm.system import FMMSystem
from ghost_backend.twod.fmm.factor import FMMFactor
from ghost_backend.execution.options import execution_scope,validate_options
from ghost_backend.twod import solver as rcs
from test_near_separation import pair_mesh
from test_experimental_cpu import fixture,fields

try:library();NATIVE=True
except (OSError,RuntimeError):NATIVE=False


@unittest.skipUnless(NATIVE,'Build the optional FMM2D library to run native qualification.')
class FMMTests(unittest.TestCase):
    def test_persistent_plan_changes_density_and_releases_native_storage(self):
        rng=np.random.default_rng(2701)
        xy=np.vstack((rng.normal(size=(120,2)),.02*rng.normal(size=(200,2))))
        normals=rng.normal(size=xy.shape)
        for k in (12.,160.-3j):
            plan=NativePlan(xy,k)
            with self.assertRaises(ValueError):plan.points.setflags(write=True)
            with self.assertRaises(AttributeError):plan.k=k+1
            for nd,gradient,dipole in ((1,False,False),(3,True,False),(17,True,True),(1,False,False)):
                c=rng.normal(size=(len(xy),nd))+1j*rng.normal(size=(len(xy),nd))
                kwargs=dict(dipoles=c,normals=normals) if dipole else dict(charges=c)
                expected=evaluate(xy,k,gradient=gradient,**kwargs)
                actual=evaluate(xy,k,gradient=gradient,plan=plan,**kwargs)
                for a,b in zip(actual,expected):
                    if a is not None:np.testing.assert_allclose(a,b,rtol=2e-8,atol=2e-8)
            if hasattr(library(),'ghost_fmm_create'):
                self.assertEqual(plan.builds,1)
                self.assertGreater(plan.bytes.value,0)
            with self.assertRaises(ValueError):evaluate(xy+.1,k,charges=c,plan=plan)
            plan.close();plan.close()
            self.assertEqual(plan.bytes.value,0)

    def test_disabling_rhs_compression_solves_every_requested_column(self):
        # Distinct nearby angles are compressible but remain valid output keys.
        result=rcs.solve_monostatic_rcs_2d(fixture('pec',32),[1.],list(np.linspace(20.,21.,19)),
            geometry_units='meters',solver_method='fmm',compute_condition_number=False,
            execution_options=dict(rhs_compression='off',fmm_pec_cfie=False))
        for factor in result['metadata']['fmm_factors']:
            self.assertEqual(factor['input_rhs_columns'],19)
            self.assertEqual(factor['solved_rhs_columns'],19)
        for pol in ('VV','HH'):
            self.assertTrue(np.all(np.isfinite(fields(result,pol))))
        reference=rcs.solve_monostatic_rcs_2d(fixture('pec',32),[1.],list(np.linspace(20.,21.,19)),
            geometry_units='meters',solver_method='direct',compute_condition_number=False)
        for pol in ('VV','HH'):
            np.testing.assert_allclose(fields(result,pol),fields(reference,pol),rtol=3e-8,atol=1e-12)

    def test_native_charge_gradient_dipole(self):
        rng=np.random.default_rng(382)
        xy=rng.random((350,2));c=rng.normal(size=(350,2))+1j*rng.normal(size=(350,2))
        normals=rng.normal(size=(350,2))
        diff=xy[:,None]-xy[None,:];r=np.linalg.norm(diff,axis=-1);np.fill_diagonal(r,1.)
        for k in (12.,12.-2j,160.):
            G=.25j*hankel2(0,k*r);np.fill_diagonal(G,0)
            H=-.25j*k*hankel2(1,k*r)/r;np.fill_diagonal(H,0)
            v,g=evaluate(xy,k,c,gradient=True)
            self.assertLess(np.linalg.norm(v-G@c)/np.linalg.norm(G@c),2e-9)
            expected=np.stack([(H*diff[:,:,d])@c for d in range(2)],axis=1)
            self.assertLess(np.linalg.norm(g-expected)/np.linalg.norm(expected),2e-9)
            expected=(-H*np.einsum('ijc,jc->ij',diff,normals))@c
            v,_=evaluate(xy,k,dipoles=c,normals=normals)
            self.assertLess(np.linalg.norm(v-expected)/np.linalg.norm(expected),2e-9)

    def test_near_weighted_masked_and_adjoint(self):
        m=pair_mesh(.9,.001,.4);k=.3-.03j
        f=GalerkinKernel(m,k)
        rng=np.random.default_rng(5);x=rng.normal(size=(len(m.nodes),3))+1j*rng.normal(size=(len(m.nodes),3))
        weights=np.array([2+3j,1-2j]);mask=np.array([1,0])
        s,kp=rcs._assemble_linear_operator_matrices(m,k,True)
        _,d=rcs._assemble_linear_operator_matrices(m,k,False)
        w=rcs._assemble_linear_hypersingular_matrix(m,k)
        for kind,a in [('S',s),('KP',kp),('K',d),('W',w)]:
            np.testing.assert_allclose(f.apply(kind,x),a@x,rtol=2e-9,atol=2e-12)
            system=FMMSystem(len(m.nodes));system.add(f,kind,weight=np.arange(len(m.nodes))+1j,mask=mask,coefficient=weights)
            # Every pair in this tiny fixture is corrected; sparse_near is the
            # independently assembled accurate operator, not its FMM evaluation.
            full=system.sparse_near().toarray()
            np.testing.assert_allclose(system@x,full@x,rtol=2e-9,atol=2e-12)
            np.testing.assert_allclose(system.H@x,full.conj().T@x,rtol=2e-9,atol=2e-12)
        combined=FMMSystem(len(m.nodes))
        for kind in ('S','KP','K','W'):combined.add(f,kind,mask=mask,coefficient=weights)
        full=combined.sparse_near().toarray()
        np.testing.assert_allclose(combined@x,full@x,rtol=2e-9,atol=2e-12)
        np.testing.assert_allclose(combined.H@x,full.conj().T@x,rtol=2e-9,atol=2e-12)

    def test_material_fields_without_dense_fallback(self):
        for kind in ('pec','ibc','lossy','mixed','layered','coated'):
            with self.subTest(kind=kind):
                snapshot=fixture(kind,32)
                args=(snapshot,[.6],[0.,37.,90.])
                a=rcs.solve_monostatic_rcs_2d(*args,geometry_units='meters')
                with patch('ghost_backend.linalg.dense.DenseFactor',side_effect=AssertionError('Dense fallback')):
                    b=rcs.solve_monostatic_rcs_2d(*args,geometry_units='meters',solver_method='fmm',
                        execution_options=dict(fmm_pec_cfie=False))
                for pol in ('VV','HH'):
                    self.assertLess(np.max(abs(fields(a,pol)-fields(b,pol)))/np.max(abs(fields(a,pol))),3e-8)
                self.assertEqual(b['metadata']['solver_method'],'galerkin_fmm_gmres')
                self.assertTrue(b['metadata']['fmm_factors'])
                for event in b['metadata']['fmm_factors']:
                    self.assertFalse(event['dense_matrix_built'])
                    self.assertEqual(event['spatial_coarse_eligible'],kind=='pec')
                    self.assertLessEqual(event['max_relative_residual'],1e-9)

    def test_iteration_failure_and_storage_rejection(self):
        m=pair_mesh(2.,.2)
        with self.assertRaises(MemoryError):GalerkinKernel(m,1.,budget=10)
        f=GalerkinKernel(m,1.)
        a=FMMSystem(len(m.nodes));a.add(f,'S')
        with execution_scope(dict(factorization='fmm',fmm_max_iterations=1,fmm_recycle_vectors=0)):
            factor=FMMFactor(a)
            with patch('ghost_backend.twod.fmm.factor.gmres',return_value=(np.zeros(len(a),complex),1)):
                with self.assertRaisesRegex(RuntimeError,'GMRES failed'):factor.solve(np.ones(len(a)))
        with self.assertRaises(ValueError):validate_options(dict(fmm_tolerance=1e-3))

    def test_combined_field_at_interior_dirichlet_modes(self):
        from ghost_backend.validation.cylinder import pec_cylinder_backscatter_amplitude
        for ka in (2.404825557695773,5.520078110286311):
            freq=ka*299792458/(2*np.pi*.06)/1e9
            result=rcs.solve_monostatic_rcs_2d_single_polarization(fixture('pec',256),[freq],[0.],
                'TM',geometry_units='meters',solver_method=' FMM ',
                execution_options=dict(fmm_pec_cfie=True))
            row=result['samples'][0];value=complex(row['rcs_amp_real'],row['rcs_amp_imag'])
            expected=pec_cylinder_backscatter_amplitude(.06,freq*1e9,'TM')
            self.assertLess(abs(value-expected)/abs(expected),2e-3)
            self.assertEqual(result['metadata']['fmm_factors'][0]['representation'],'combined_field_pec')
            self.assertFalse(result['metadata']['fmm_factors'][0]['spatial_coarse_eligible'])

    def test_certified_path_and_frequency_cache_lifetime(self):
        result=rcs.solve_monostatic_rcs_2d_certified(fixture('pec',32),[.6,.8],[0.],
            geometry_units='meters',solver_method='fmm')
        self.assertTrue(result['metadata']['mesh_convergence_certified'])
        self.assertTrue(result['metadata']['condition_est_computed'])
        self.assertFalse(result['metadata']['fmm_factors'][0]['approximation_error_certified'])

    def test_impedance_sheet_and_sheet_with_pec(self):
        from test_thin_sheet import sheet_snapshot
        for mixed in (False,True):
            snapshot=sheet_snapshot([[-.12,-.09],[.12,-.09]],24)
            if mixed:snapshot['segments']+=fixture('pec',24)['segments']
            args=(snapshot,[1.],[30.,60.,90.])
            a=rcs.solve_monostatic_rcs_2d(*args,geometry_units='meters')
            b=rcs.solve_monostatic_rcs_2d(*args,geometry_units='meters',solver_method='fmm')
            self.assertTrue(all(not e['spatial_coarse_eligible'] for e in b['metadata']['fmm_factors']))
            for pol in ('VV','HH'):
                self.assertLess(np.max(abs(fields(a,pol)-fields(b,pol)))/np.max(abs(fields(a,pol))),3e-8)

    def test_dielectric_at_twenty_ghz(self):
        args=(fixture('lossy',256),[20.],[31.])
        a=rcs.solve_monostatic_rcs_2d(*args,geometry_units='meters')
        b=rcs.solve_monostatic_rcs_2d(*args,geometry_units='meters',solver_method='fmm')
        for pol in ('VV','HH'):
            self.assertLess(np.max(abs(fields(a,pol)-fields(b,pol)))/np.max(abs(fields(a,pol))),3e-8)

    def test_factor_lifetime_does_not_require_cyclic_gc(self):
        import weakref
        f=GalerkinKernel(pair_mesh(2.,.2),1.)
        a=FMMSystem(f.n);a.add(f,'S')
        factor=FMMFactor(a);reference=weakref.ref(factor)
        del factor
        self.assertIsNone(reference())

if __name__=='__main__':unittest.main()
