"""Qualification of fused CFIE, guarded quadrature and bounded sweep reuse."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest

sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parent)]
from ghost_backend.twod import solver as s
from ghost_backend.twod.fmm import galerkin
from ghost_backend.twod.fmm.system import FMMSystem
from ghost_backend.twod.fmm.factor import FMMFactor
from ghost_backend.twod.fmm.quadrature import quadrature_order
from ghost_backend.twod.fmm.memory import forecast,rhs_batch_size
from ghost_backend.execution.options import execution_scope,validate_options
from ghost_backend.execution.policy import native_fmm_available
from ghost_backend.twod.preparation import prepare_geometry
from general_fixtures import fixture
from test_experimental_cpu import fixture as circle_fixture,fields


def mesh_for(case,n,freq):
    snapshot=fixture(case,n)
    _,_,materials,scale=prepare_geometry(snapshot,None,'meters')
    wavelength,_,_=s._mesh_wavelength_for_snapshot(snapshot,materials,freq)
    panels=s._build_panels(snapshot,scale,wavelength,max_panels=100000)
    k=2*np.pi*freq*1e9/s.C0
    infos=s._build_coupled_panel_info(panels,materials,freq,'TM',k)
    mesh,_=s._build_linear_mesh_interface_aware(panels,infos)
    return mesh,k


def test_quadrature_and_saved_option_guards():
    assert quadrature_order(1.,1.)[0]==6
    assert quadrature_order(1.,1.01)[0]==8
    assert quadrature_order(1.,.1,eps=1e-12)[0]==8
    assert quadrature_order(1.,.1,requested=8)[0]==8
    assert quadrature_order(30.,1.,requested=6)[0]>=19
    with pytest.raises(ValueError):quadrature_order(130.,1.)
    for key,values in [('fmm_quadrature_order',[True,4,65,6.5]),
                       ('fmm_recycle_vectors',[True,-1,65,'yes']),('fmm_pec_cfie',[1,'yes'])]:
        for value in values:
            with pytest.raises(ValueError):validate_options({key:value})
    old=validate_options(dict(fmm_pec_cfie=False,fmm_recycle_vectors=12))
    assert old['fmm_pec_cfie'] is False and old['fmm_recycle_vectors']==12


def test_memory_forecast_counts_geometry_not_storage_cap():
    resources=dict(fmm_geometry=dict(panels=256,kernels=1,near_pairs=2000,
                                     max_electrical_panel_length=.1))
    peaks=[]
    for cap in (64,2048):
        with execution_scope(dict(factorization='fmm',compressed_storage_mib=cap)):
            plan=forecast(256,256,1,181,cap*1024**2,resources)
            peaks.append(plan['peak_bytes'])
            assert plan['geometry_preflight']
            assert plan['angle_batch_size']==181
    assert peaks[0]==peaks[1] and peaks[0]<.3*1024**3
    with execution_scope(dict(ram_budget_gib=.25)):
        assert rhs_batch_size(8192,256)<32


def test_capped_near_forecast_keeps_positive_workspace_components():
    resources=dict(fmm_geometry=dict(panels=100000,kernels=1,near_pairs=8000,
        near_count_capped=True,max_electrical_panel_length=.1))
    plan=forecast(100000,100000,1,32,16*1024**2,resources)
    assert plan['near_count_capped']
    assert all(value>0 for value in plan['components'].values())
    assert plan['peak_bytes']==sum(plan['components'].values())


native=pytest.mark.skipif(not native_fmm_available(),reason='Native FMM is optional')


@native
@pytest.mark.parametrize('degree',[2,3])
def test_combined_field_preserves_polynomial_forward_and_adjoint(degree):
    with execution_scope(dict(basis_order=degree)):
        mesh,k=mesh_for('rectangle',24,1.)
        f=galerkin.GalerkinKernel(mesh,k)
        assert f.order>=8
        a=FMMSystem(f.n);eta=-1j*k;a.add_combined_field(f,eta)
        try:
            eye=np.eye(f.n,dtype=complex)
            reference=f.apply('K',eye)+eta*f.apply('S',eye)
            rng=np.random.default_rng(920+degree)
            x=rng.normal(size=(f.n,17))+1j*rng.normal(size=(f.n,17))
            np.testing.assert_allclose(a@x,reference@x,rtol=2e-10,atol=2e-11)
            np.testing.assert_allclose(a.H@x,reference.conj().T@x,rtol=2e-10,atol=2e-11)
        finally:f.native_plan.close()


@native
@pytest.mark.parametrize('case',['rectangle','reentrant','acute','gap','dielectric','mixed','sheet'])
@pytest.mark.parametrize('freq',[3.,20.])
def test_guarded_quadrature_matches_dense_operators(case,freq):
    mesh,k=mesh_for(case,384 if case=='mixed' and freq==20. else 192,freq)
    if case=='dielectric':k*=np.sqrt(3-.1j)
    f=galerkin.GalerkinKernel(mesh,k)
    rng=np.random.default_rng(957)
    x=rng.normal(size=(f.n,3))+1j*rng.normal(size=(f.n,3))
    with execution_scope(dict(factorization='dense')):
        S,KP=s._assemble_linear_operator_matrices(mesh,k,True)
        _,K=s._assemble_linear_operator_matrices(mesh,k,False,compute_single_layer=False)
        W=s._assemble_linear_hypersingular_matrix(mesh,k)
    for kind,a in [('S',S),('KP',KP),('K',K),('W',W)]:
        expected=a@x
        assert np.linalg.norm(f.apply(kind,x)-expected)/max(np.linalg.norm(expected),1e-20)<3e-10
    assert np.shares_memory(f.near['K'].data,f.near['KP'].data)
    f.native_plan.close()


@native
def test_fused_cfie_forward_adjoint_and_wide_rhs():
    mesh,k=mesh_for('reentrant',96,3.)
    f=galerkin.GalerkinKernel(mesh,k)
    fused=FMMSystem(f.n);fused.add_combined_field(f,-1j*k)
    separate=FMMSystem(f.n);separate.add(f,'K');separate.add(f,'S',weight=-1j*k)
    rng=np.random.default_rng(192)
    x=rng.normal(size=(f.n,17))+1j*rng.normal(size=(f.n,17))
    for a,b in [(fused@x,separate@x),(fused.H@x,separate.H@x)]:
        np.testing.assert_allclose(a,b,rtol=2e-9,atol=2e-12)
    y=rng.normal(size=f.n)+1j*rng.normal(size=f.n)
    assert abs(np.vdot(y,fused@x[:,0])-np.vdot(fused.H@y,x[:,0]))<1e-10
    with patch.object(galerkin,'evaluate',wraps=galerkin.evaluate) as evaluate:
        fused@x
        assert all(call.kwargs['charges'].shape[1]<=8 for call in evaluate.call_args_list)
        assert all(call.kwargs['dipoles'] is not None for call in evaluate.call_args_list)
    f.native_plan.close()


@native
def test_rhs_cache_reuses_columns_and_checks_unscaled_residual():
    mesh,k=mesh_for('rectangle',64,1.)
    f=galerkin.GalerkinKernel(mesh,k)
    a=FMMSystem(f.n);a.add(f,'S')
    rng=np.random.default_rng(922)
    basis=rng.normal(size=(f.n,4))+1j*rng.normal(size=(f.n,4))
    b=basis@(rng.normal(size=(4,19))+1j*rng.normal(size=(4,19)))
    with execution_scope(dict(factorization='fmm')):
        factor=FMMFactor(a)
        first=factor.solve(b)
        count=factor.details['solved_rhs_columns']
        second=factor.solve(b@np.eye(19)[:,::-1])
        assert factor.details['solved_rhs_columns']==count
        assert factor.details['rhs_basis_reused_batches']==1
        np.testing.assert_allclose(second,first[:,::-1],rtol=1e-8,atol=1e-9)
        assert np.max(factor.relative_residual)<=factor.tolerance
        assert factor.details['rhs_basis_columns']<=factor.rhs_basis_capacity
    f.native_plan.close()


@native
def test_default_cfie_matches_analytic_high_frequency_circle():
    from ghost_backend.validation.cylinder import pec_cylinder_backscatter_amplitude
    freq=30*299792458/(2*np.pi*.06)/1e9
    result=s.solve_monostatic_rcs_2d_single_polarization(circle_fixture('pec',512),[freq],
        [0.,37.,90.],'TM',geometry_units='meters',solver_method='fmm',compute_condition_number=False)
    expected=pec_cylinder_backscatter_amplitude(.06,freq*1e9,'TM')
    values=np.array([complex(r['rcs_amp_real'],r['rcs_amp_imag']) for r in result['samples']])
    assert np.max(abs(values-expected))/abs(expected)<.002
    factor=result['metadata']['fmm_factors'][0]
    assert factor['representation']=='combined_field_pec' and factor['combined_field_fused']
    assert factor['iterative_method']=='gmres'
