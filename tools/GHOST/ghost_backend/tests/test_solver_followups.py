"""Numerical and API regressions for automatic solver follow-ups."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parent)]
from ghost_backend.twod.assembly.kernels import KernelTable, values
from ghost_backend.twod.assembly.native.table import library
from ghost_backend.twod import solver
from general_fixtures import fixture


@pytest.mark.parametrize('k,upper',[(.3-.03j,2.),(80-2j,.3),(1600-900j,.14)])
def test_table_evaluator_against_special_functions_and_fallback(k,upper):
    table=KernelTable(k,upper)
    rng=np.random.default_rng(3919)
    r=np.r_[table.bounds,np.geomspace(5e-13,upper,10001),rng.uniform(5e-13,upper,10000)]
    ref=values(k,r)
    for channel in (-1,0,1):
        expected=ref if channel==-1 else ref[:,channel]
        got=table.evaluate(r,channel)
        assert np.max(abs(got-expected)/np.maximum(abs(expected),1e-280))<2e-13
    if library() is not None:assert table.native_checked
    np.testing.assert_allclose(table.evaluate(upper/2),values(k,upper/2),rtol=2e-13,atol=1e-280)
    np.testing.assert_allclose(table.evaluate(r[::2]),ref[::2],rtol=2e-13,atol=1e-280)
    with patch('ghost_backend.twod.assembly.native.table.library',return_value=None):
        np.testing.assert_allclose(table.evaluate(r),ref,rtol=2e-13,atol=1e-280)
    assert np.isnan(table.evaluate(np.array([0.,upper*2,np.nan]))).all()


@pytest.mark.parametrize('method',['auto','direct','experimental_cpu'])
def test_numpy_public_input_sequences(method):
    args=dict(geometry_snapshot=fixture('rectangle',32),geometry_units='meters',solver_method=method)
    expected=solver.solve_monostatic_rcs_2d(frequencies_ghz=[1.],elevations_deg=[0.,90.],**args)
    got=solver.solve_monostatic_rcs_2d(frequencies_ghz=np.array([1.]),elevations_deg=np.array([0.,90.]),**args)
    assert got['co_solved_samples']==expected['co_solved_samples']
    with pytest.raises(ValueError,match='one-dimensional'):
        solver.solve_monostatic_rcs_2d(frequencies_ghz=[1.],elevations_deg=np.array([[0.,90.]]),**args)


def test_numpy_bistatic_sequences():
    result=solver.solve_bistatic_rcs_2d(fixture('rectangle',24),np.array([1.]),
        np.array([0.,90.]),np.array([0.,90.]),geometry_units='meters')
    assert len(result['co_solved_samples']['VV'])==4


def test_fmm_forecast_accounts_for_material_quadrature_fill_and_coarse_work():
    from ghost_backend.twod.fmm.memory import forecast
    resources=dict(panels=1000,geometric_near_pairs=7000,fmm_kernel_orders=[8])
    a=forecast(1000,2000,1,32,64*1024**2,resources)
    b=forecast(1000,2000,2,32,64*1024**2,dict(resources,fmm_kernel_orders=[8,32]))
    assert a['peak_bytes']==sum(a['components'].values())
    assert b['peak_bytes']>a['peak_bytes']
    assert a['components']['sparse_preconditioner_peak_bytes']>0
    assert a['components']['coarse_workspace_bytes']>0
    assert b['quadrature_points']==40000


def test_spatial_coarse_preconditioner_and_adjoint():
    from scipy.sparse import eye
    from scipy.sparse.linalg import LinearOperator
    from ghost_backend.twod.fmm.factor import FMMFactor
    class Operator(LinearOperator):
        def __init__(self,a):
            super().__init__(dtype=np.dtype(complex),shape=a.shape)
            self.matrix=a; self.kernels=[]
        def __len__(self):return len(self.matrix)
        def _matvec(self,x):return self.matrix@x
        def _rmatvec(self,x):return self.matrix.conj().T@x
        def sparse_near(self):return eye(len(self),dtype=complex,format='csc')
        def report(self):return {}
    rng=np.random.default_rng(98);n=128
    u=rng.normal(size=(n,4))+1j*rng.normal(size=(n,4))
    a=np.eye(n)+.01*u@u.conj().T
    factor=FMMFactor(Operator(a),coordinates=rng.random((n,2)))
    x=rng.normal(size=n)+1j*rng.normal(size=n)
    y=rng.normal(size=n)+1j*rng.normal(size=n)
    factor.solve(x)
    previous=factor.pre,factor.preh,list(factor.recycled)
    assert factor._coarse_correction()
    assert factor.recycled and all(product is None for _,product in factor.recycled)
    np.testing.assert_allclose(np.vdot(y,factor.pre@x),np.vdot(factor.preh@y,x),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(a@factor.solve(x),x,rtol=1e-8,atol=1e-9)
    # With no remaining illumination, a trial cannot repay setup. Its solution
    # remains valid, and both original preconditioners and cached products return.
    assert factor.pre is previous[0] and factor.preh is previous[1]
    assert not factor.details['coarse_trial_accepted']
    np.testing.assert_allclose(a@factor.solve(y),y,rtol=1e-8,atol=1e-9)


def test_measured_history_needs_repetition_and_respects_worker_ram(tmp_path,monkeypatch):
    from ghost_backend.execution import timing_history as history
    monkeypatch.setenv('GHOST_TIMING_CACHE_DIR',str(tmp_path))
    choice=dict(selected='dense',retry_order=['compressed'],admission_budget_gib=8,
        candidates=dict(dense=dict(cost=10.,peak_gb=4.),compressed=dict(cost=14.,peak_gb=1.)))
    valid=dict(quality_gate=dict(passed=True))
    for mode,t in [('dense',5.),('compressed',2.)]:history.record('case',mode,t,valid)
    assert history.adjust(choice,'case')['selected']=='dense'
    for mode,t in [('dense',5.),('compressed',2.)]:history.record('case',mode,t,valid)
    revised=history.adjust(choice,'case')
    assert revised['selected']=='compressed'
    assert revised['timing_model']=='identical_request_host_median_v1'
    assert history.adjust(choice,'other request')['selected']=='dense'
    choice['selected']='compressed'
    for _ in range(5):history.record('case','dense',.1,valid)
    assert history.adjust(choice,'case',batch=True)['selected']=='compressed'
    history.record('bad','fmm',1.,dict(quality_gate=dict(passed=False)))
    assert 'bad' not in history.read()
    history.cache_path().write_text('{broken')
    assert history.adjust(choice,'case')['selected']=='compressed'


def test_timing_request_identity_changes_with_angles_precision_and_settings():
    from ghost_backend.execution.timing_history import request_key
    from ghost_backend.execution.options import validate_options
    from ghost_backend.linalg.refined_lu import linear_precision
    options=validate_options({})
    arguments=dict(geometry_snapshot=fixture('rectangle',24),frequencies_ghz=[1.],
        elevations_deg=[0.,90.],geometry_units='meters')
    base=request_key(arguments,options,'solve_monostatic_rcs_2d')
    assert len(base)==64
    assert request_key(dict(arguments,elevations_deg=[0.,45.]),options,'solve_monostatic_rcs_2d')!=base
    assert request_key(arguments,dict(options,fmm_solver_tolerance=1e-10),'solve_monostatic_rcs_2d')!=base
    with linear_precision('mixed'):
        assert request_key(arguments,options,'solve_monostatic_rcs_2d')!=base


def test_unwritable_timing_cache_returns_after_one_attempt(tmp_path,monkeypatch):
    from ghost_backend.execution import timing_history as history
    monkeypatch.setenv('GHOST_TIMING_CACHE_DIR',str(tmp_path))
    attempts=[]
    def denied(*args,**kwargs):
        attempts.append(args)
        raise PermissionError('read-only timing cache')
    monkeypatch.setattr(history,'open',denied,raising=False)
    history.record('completed','dense',1.,dict(quality_gate=dict(passed=True)))
    assert len(attempts)==1
    assert not history.cache_path().exists()
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('failure', [None, 'write', 'replace', 'cleanup'])
def test_timing_cache_cleanup_with_legacy_unlink(tmp_path, monkeypatch, failure):
    from ghost_backend.execution import timing_history as history
    monkeypatch.setenv('GHOST_TIMING_CACHE_DIR', str(tmp_path))
    valid = dict(quality_gate=dict(passed=True))
    history.record('previous', 'dense', 1., valid)
    previous = history.cache_path().read_bytes()
    attempts = []

    class LegacyPath(type(tmp_path)):
        # Match Python 3.6's signature so newer keyword arguments fail here.
        def unlink(self):
            attempts.append(self)
            if failure == 'cleanup':
                raise PermissionError('temporary cleanup denied')
            return super().unlink()

    def denied(*args, **kwargs):
        raise PermissionError('optional cache write denied')

    monkeypatch.setattr(history, 'Path', LegacyPath)
    if failure == 'write':
        monkeypatch.setattr(history.json, 'dump', denied)
    if failure in ('replace', 'cleanup'):
        monkeypatch.setattr(history.os, 'replace', denied)
    history.record('completed', 'dense', 2., valid)
    assert len(attempts) == 1
    if failure is None:
        assert set(history.read()) == {'previous', 'completed'}
    else:
        assert history.cache_path().read_bytes() == previous
    assert len(list(tmp_path.glob('*.tmp'))) == (1 if failure == 'cleanup' else 0)


def test_one_frequency_reference_fallback_does_not_train_timings(tmp_path,monkeypatch):
    from ghost_backend.execution import timing_history as history
    monkeypatch.setenv('GHOST_TIMING_CACHE_DIR',str(tmp_path))
    history.record('mixed','dense',1.,dict(quality_gate=dict(passed=True),frequency_metadata=[
        dict(metadata=dict(backend_selection=dict(selected='dense'),adaptive_mesh=dict(fallback=False))),
        dict(metadata=dict(backend_selection=dict(selected='dense'),adaptive_mesh=dict(fallback=True)))]))
    assert not history.cache_path().exists()


def test_resident_matrix_credit_preserves_absolute_budgets():
    from ghost_backend.execution.options import execution_scope
    with patch.object(solver,'_detect_available_gb',return_value=4.):
        with execution_scope(dict(ram_budget_gib=4.2)):
            assert solver._solve_memory_limit_gb()==3.6
            assert solver._solve_memory_limit_gb(resident_gb=1.)==4.2
        with execution_scope(dict(ram_budget_gib=8.),memory_budget_gib=3.):
            assert solver._solve_memory_limit_gb(resident_gb=1.)==3.


def test_resident_credit_requires_the_exact_reusable_owned_matrix():
    from ghost_backend.twod.assembly import session
    from ghost_backend.execution.options import execution_scope
    mesh=object();infos=[];state=session.AssemblySession()
    matrix=np.zeros((16,16),complex)
    state.pending=('matching',(matrix,None))
    with execution_scope(dict(factorization='dense')),session._SESSION.override(state), \
         patch.object(session,'system_key',return_value='matching'):
        assert session.reusable_dense_bytes(mesh,infos,'TM','multi_region',16)==matrix.nbytes
        assert session.reusable_dense_bytes(mesh,infos,'TE','multi_region',16)==0
        assert session.reusable_dense_bytes(mesh,infos,'TM','multi_region',15)==0
        state.pending=('different',(matrix,None))
        assert session.reusable_dense_bytes(mesh,infos,'TM','multi_region',16)==0
        state.pending=('matching',(matrix.view(),None))
        assert session.reusable_dense_bytes(mesh,infos,'TM','multi_region',16)==0
