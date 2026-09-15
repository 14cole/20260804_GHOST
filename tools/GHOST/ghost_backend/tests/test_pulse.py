"""Physical accuracy, matrix-free adjoints, saved controls and pulse backends."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
from scipy.special import hankel2
from scipy.integrate import quad_vec

sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parent)]
from ghost_backend.execution.options import execution_scope,validate_options,validate_for_run
from ghost_backend.execution.policy import native_fmm_available
from ghost_backend.twod import solver as s
from ghost_backend.twod.pulse.kernel import PulseKernel,PulseSystem
from ghost_backend.twod.pulse.runtime import PulseOracle,dense_matrix
from ghost_backend.twod.pulse.coefficients import accurate_pairs,blocks,self_single_layer
from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
from test_near_separation import pair_mesh
from test_fmm_efficiency import mesh_for
from test_experimental_cpu import fixture,fields
from general_fixtures import fixture as polygon
from ghost_backend.validation.cylinder import (
    pec_cylinder_backscatter_amplitude,sigma_dielectric_cylinder,sigma_impedance_cylinder)


def solve(snapshot,mode='dense',angles=(0.,37.,90.),frequency=1.,**options):
    return s.solve_monostatic_rcs_2d(snapshot,[frequency],list(angles),geometry_units='meters',
        solver_method='experimental_cpu',compute_condition_number=False,
        execution_options=dict(discretization='pulse',factorization=mode,**options))


def test_profiles_and_unsupported_routes_are_explicit():
    from ghost_backend.runs.setup import validate_setup
    from test_execution_options import setup_record
    profile=validate_options(dict(discretization='pulse',pulse_pec_cfie=False))
    assert validate_setup(setup_record(profile))['execution_options']==profile
    for opts in ({'discretization':'typo'},{'pulse_pec_cfie':'auto'}):
        with pytest.raises(ValueError):validate_options(opts)
    for kind,scattering,precision in [('bor','monostatic','double'),('2d','bistatic','double'),('2d','monostatic','mixed')]:
        with pytest.raises(ValueError,match='Pulse'):
            validate_for_run(profile,kind=kind,scattering=scattering,precision=precision)
    with pytest.raises(ValueError,match='Thin sheets'):
        solve(polygon('sheet',24))


def test_pulse_uses_panel_refinement_with_adaptive_gui_defaults():
    from ghost_backend.execution.options import efficient_defaults
    profile=validate_options(dict(efficient_defaults(),discretization='pulse'))
    assert profile['mesh_strategy']=='global'
    assert profile['basis_order']==1
    with pytest.raises(ValueError,match='polynomial enrichment'):
        validate_options(dict(discretization='pulse',basis_order=2))


@pytest.mark.parametrize('k',[.3,12.-2j])
def test_near_coefficients_match_independent_adaptive_integrals(k):
    mesh=pair_mesh(.9,1e-5,.4);g=AssemblyGeometry(mesh)
    result=accurate_pairs(g,k,np.array([0]),np.array([1]),{'S','KP','K'})
    x=g.centers[0];e=mesh.elements[1]
    def integrand(t):
        delta=x-(e.p0+t*(e.p1-e.p0));r=np.linalg.norm(delta)
        G=.25j*hankel2(0,k*r)
        grad=-.25j*k*hankel2(1,k*r)*delta/r
        return e.length*np.array([G,grad@g.normals[0],-grad@g.normals[1]])
    projection=np.clip(np.dot(x-e.p0,e.p1-e.p0)/e.length**2,0,1)
    expected,error=quad_vec(integrand,0,1,points=[projection],epsabs=1e-11,epsrel=1e-11)
    actual=np.array([result[key][0] for key in ['S','KP','K']])
    np.testing.assert_allclose(actual,expected,rtol=2e-9,atol=2e-11)


native=pytest.mark.skipif(not native_fmm_available(),reason='Native FMM is optional')


@pytest.mark.parametrize('ratio,electrical,order',[(6.,.6,4),(16.,.15,3),(200.,.01,2)])
@pytest.mark.parametrize('phase',[0.,-.3])
def test_far_grading_boundaries_match_independent_integrals(ratio,electrical,order,phase):
    from ghost_backend.twod.pulse import coefficients
    # Step just inside each guard so roundoff cannot select the full rule.
    ratio*=1+1e-9;electrical*=1-1e-9
    for direction in [.1,.7,1.4]:
        mesh=pair_mesh(ratio*np.cos(direction),ratio*np.sin(direction),.47)
        g=AssemblyGeometry(mesh);k=electrical*np.exp(1j*phase)
        with patch.object(coefficients,'point_pairs',wraps=coefficients.point_pairs) as point:
            result=blocks(g,k,[0],[1],{'S','KP','K'},8,graded=True)
        assert point.call_args.args[-1]==order
        e=mesh.elements[1];x=g.centers[0]
        def integrand(t):
            delta=x-e.p0-t*(e.p1-e.p0);r=np.linalg.norm(delta)
            G=.25j*hankel2(0,k*r)
            gradient=-.25j*k*hankel2(1,k*r)*delta/r
            return e.length*np.array([G,gradient@g.normals[0],-gradient@g.normals[1]])
        expected,_=quad_vec(integrand,0,1,epsabs=1e-14,epsrel=1e-13)
        actual=np.array([result[key][0,0] for key in ['S','KP','K']])
        np.testing.assert_allclose(actual,expected,rtol=3e-10,atol=1e-15)


@pytest.mark.parametrize('k',[.3,12.-2j])
def test_self_integral_cache_preserves_actual_panel_length(k):
    for length in [.12345678901249,.12345678901251,1.0000000000049,1.0000000000051]:
        expected,_=quad_vec(lambda t:.5j*length*hankel2(0,k*length*t),
                            0,.5,epsabs=1e-14,epsrel=1e-13)
        np.testing.assert_allclose(self_single_layer(k,length),expected,rtol=2e-12,atol=1e-15)


@native
def test_forward_adjoint_masks_and_cfie_match_explicit_collocation():
    mesh,k=mesh_for('reentrant',64,3.)
    f=PulseKernel(mesh,k);a=PulseSystem(f.n);rng=np.random.default_rng(42)
    for kind in ['S','KP','K']:
        a.add(f,kind,weight=rng.normal(size=f.n)+1j*rng.normal(size=f.n),
              mask=np.arange(f.n)%2,coefficient=rng.normal(size=f.n)+1j*rng.normal(size=f.n))
    dense=dense_matrix(PulseOracle(a),lambda:None)
    x=rng.normal(size=(f.n,17))+1j*rng.normal(size=(f.n,17))
    for actual,expected in [(a@x,dense@x),(a.H@x,dense.conj().T@x)]:
        np.testing.assert_allclose(actual,expected,rtol=2e-9,atol=2e-11)
    fused=PulseSystem(f.n);fused.add_combined_field(f,-1j*k)
    dense=dense_matrix(PulseOracle(fused),lambda:None)
    np.testing.assert_allclose(fused@x,dense@x,rtol=2e-9,atol=2e-11)
    np.testing.assert_allclose(fused.H@x,dense.conj().T@x,rtol=2e-9,atol=2e-11)
    f.native_plan.close()


@native
@pytest.mark.parametrize('case',['pec','ibc','lossy','mixed','layered','coated'])
def test_dense_compressed_fmm_share_the_same_pulse_equation(case):
    snapshot=fixture(case,48)
    with patch.object(s,'_assemble_linear_operator_matrices',side_effect=AssertionError('Galerkin assembly')):
        reference=solve(snapshot)
        compressed=solve(snapshot,'compressed')
        with patch('ghost_backend.linalg.dense.DenseFactor',side_effect=AssertionError('Dense FMM fallback')):
            fmm=solve(snapshot,'fmm')
    for result in [compressed,fmm]:
        assert result['metadata']['discretization']=='pulse'
        for pol in ['VV','HH']:
            scale=np.max(abs(fields(reference,pol)))
            assert np.max(abs(fields(result,pol)-fields(reference,pol)))/scale<3e-8
    assert fmm['metadata']['solver_method']=='pulse_fmm_gmres'
    assert all(r['discretization']=='pulse_collocation' for r in fmm['metadata']['fmm_factors'])


@pytest.mark.parametrize('ka',[2.404825557695773,5.520078110286311,10.,30.])
def test_pec_tm_combined_field_matches_complex_analytic_circle(ka):
    freq=ka*s.C0/(2*np.pi*.06)/1e9
    result=solve(fixture('pec',512),frequency=freq)
    expected=pec_cylinder_backscatter_amplitude(.06,freq*1e9,'TM')
    assert np.max(abs(fields(result,'HH')-expected))/abs(expected)<.002


def test_te_and_dielectric_refinement_against_independent_series():
    # Point testing has a different convergence rate; assert its actual rate,
    # never assume that P0 and P1 have equal errors at the same panel count.
    for case in ['pec','lossy','ibc']:
        errors=[]
        for n in [128,256]:
            result=solve(fixture(case,n),angles=(0.,))
            per_pol=[]
            for label,pol in [('VV','TE'),('HH','TM')]:
                if case=='pec':
                    expected=abs(pec_cylinder_backscatter_amplitude(.06,1e9,pol))**2/(4*2*np.pi*1e9/s.C0)
                elif case=='lossy':expected=sigma_dielectric_cylinder(.06,3-.1j,1.,1e9,pol)
                else:expected=sigma_impedance_cylinder(.06,50-10j,1e9,pol)
                got=result['co_solved_samples'][label][0]['rcs_linear']
                per_pol.append(abs(got-expected)/expected)
            errors.append(np.array(per_pol))
        assert np.all(errors[1]<.65*errors[0])
        assert np.max(errors[1])<(.03 if case=='lossy' else .008)


@native
def test_many_angles_reuse_condition_and_certification():
    options=dict(discretization='pulse',factorization='fmm',angle_batch_size=16)
    snapshot=fixture('pec',48)
    result=s.solve_monostatic_rcs_2d_certified(snapshot,[.6],list(np.linspace(20.,21.,33)),
        geometry_units='meters',solver_method='fmm',execution_options=options)
    assert result['metadata']['discretization']=='pulse'
    assert result['metadata']['mesh_convergence_certified']
    assert result['metadata']['condition_est_computed']
    assert any(f['rhs_basis_reused_batches'] for f in result['metadata']['fmm_factors'])


def test_gui_discretization_roundtrip():
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from ghost_backend.ui.execution import ExecutionOptionsWidget
    app=QApplication.instance() or QApplication([])
    widget=ExecutionOptionsWidget()
    try:
        widget.set_value(dict(discretization='pulse',factorization='adaptive'))
        assert widget.basis_combo.currentData()=='pulse'
        assert widget.value()['discretization']=='pulse'
        widget.set_value({})
        assert widget.basis_combo.currentData()=='galerkin'
    finally:widget.close()
