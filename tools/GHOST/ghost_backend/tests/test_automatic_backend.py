"""Automatic selection, physical equivalence, retries and execution-node policy."""
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest

sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parent)]
from ghost_backend.execution.options import validate_options,execution_scope,current_options
from ghost_backend.execution.policy import rank_candidates,native_fmm_available
from ghost_backend.execution.selection import select_backend
from ghost_backend.execution.errors import BackendNumericalError
from ghost_backend.runs.batch import select_batch_backends
from ghost_backend.twod import solver
from general_fixtures import fixture
from test_experimental_cpu import fields


@pytest.mark.parametrize('case',['rectangle','reentrant','acute','gap','dielectric','mixed','sheet'])
def test_default_auto_matches_reference_on_arbitrary_geometry(case):
    args=(fixture(case,64),[1.],[0.,37.,90.])
    reference=solver.solve_monostatic_rcs_2d(*args,geometry_units='meters',solver_method='direct')
    result=solver.solve_monostatic_rcs_2d(*args,geometry_units='meters')
    meta=result['metadata']
    assert meta['solver_method_requested']=='auto'
    assert meta['backend_selection']['selected']=='dense'
    assert not meta['backend_selection']['optimality_guaranteed']
    for pol in ('VV','HH'):
        np.testing.assert_allclose(fields(result,pol),fields(reference,pol),rtol=3e-9,atol=1e-12)
    assert current_options() is None


def test_rank_accounts_for_speed_memory_and_unavailable_native():
    candidates=dict(dense=dict(cost=1.,peak_gb=4.),compressed=dict(cost=2.,peak_gb=2.),fmm=dict(cost=3.,peak_gb=.5))
    with patch('ghost_backend.execution.policy.native_fmm_available',return_value=True):
        assert rank_candidates(candidates,8)[0]=='dense'
        assert rank_candidates(candidates,3)[0]=='compressed'
        assert rank_candidates(candidates,1)[0]=='fmm'
    with patch('ghost_backend.execution.policy.native_fmm_available',return_value=False):
        with pytest.raises(MemoryError,match='No compatible backend fits'):rank_candidates(candidates,1)


def test_planner_forecasts_actual_polygon_mesh_without_coefficients():
    args=dict(geometry_snapshot=fixture('reentrant',96),frequencies_ghz=[1.,3.],elevations_deg=[0.,90.],
              geometry_units='meters',solver_method='auto',max_panels=10000)
    with patch('ghost_backend.compressed.memory.geometry_storage',side_effect=AssertionError('sampled coefficients')):
        result=select_backend(args,validate_options(dict(factorization='adaptive',compressed_storage_mib=64)),certified=True)
    assert len(result['meshes'])==8
    assert result['selected']=='dense'
    assert {'base','fine'}=={r['phase'] for r in result['meshes']}


@pytest.mark.skipif(not native_fmm_available(),reason='Native FMM is optional')
def test_auto_numerical_retry_preserves_accuracy_and_explicit_fmm_does_not_retry():
    args=(fixture('reentrant',48),[1.],[0.,90.])
    choice=dict(requested='adaptive',selected='fmm',retry_order=['dense'],reason='test fixture')
    with patch('ghost_backend.execution.selection.select_backend',return_value=choice), \
         patch('ghost_backend.twod.fmm.factor.FMMFactor.solve',side_effect=BackendNumericalError('injected convergence failure')):
        result=solver.solve_monostatic_rcs_2d(*args,geometry_units='meters')
        with pytest.raises(BackendNumericalError):
            solver.solve_monostatic_rcs_2d(*args,geometry_units='meters',solver_method='fmm')
    reference=solver.solve_monostatic_rcs_2d(*args,geometry_units='meters',solver_method='direct')
    assert result['metadata']['backend_selection']['selected']=='dense'
    assert result['metadata']['backend_selection']['initial_selection']=='fmm'
    assert len(result['metadata']['backend_selection']['failed_attempts'])==1
    assert result['metadata']['execution_options']['factorization']=='dense'
    for pol in ('VV','HH'):np.testing.assert_allclose(fields(result,pol),fields(reference,pol),rtol=1e-8,atol=1e-12)


@pytest.mark.skipif(not native_fmm_available(),reason='Native FMM is optional')
def test_cancellation_is_never_retried():
    choice=dict(requested='adaptive',selected='fmm',retry_order=['dense'],reason='test fixture')
    with patch('ghost_backend.execution.selection.select_backend',return_value=choice), \
         patch('ghost_backend.twod.fmm.factor.FMMFactor.solve',side_effect=InterruptedError('canceled')), \
         patch('ghost_backend.linalg.dense.DenseFactor',side_effect=AssertionError('unexpected retry')):
        with pytest.raises(InterruptedError):
            solver.solve_monostatic_rcs_2d(fixture('rectangle',48),[1.],[0.],geometry_units='meters')


def test_hpc_fmm_choice_and_retry_stay_inside_the_unit_memory_reservation():
    records=[dict(unit=str(i),backend_candidates=dict(dense=dict(cost=10.,peak_gb=7.),
             compressed=dict(cost=14.,peak_gb=4.),fmm=dict(cost=16.,peak_gb=1.))) for i in range(4)]
    with patch('ghost_backend.execution.policy.native_fmm_available',return_value=True):
        choices,summary=select_batch_backends(records,4,4,4,validate_options(dict(assembly_threads=1)))
    assert summary['fmm_units']==4
    assert all(v['selected']=='fmm' and not v['retry_order'] for v in choices.values())


def test_native_fmm_is_checked_on_execution_node():
    record=dict(unit='a',backend_candidates=dict(dense=dict(cost=10.,peak_gb=2.),fmm=dict(cost=1.,peak_gb=1.)))
    with patch('ghost_backend.execution.policy.native_fmm_available',return_value=False):
        choices,_=select_batch_backends([record],1,1,4,validate_options({}))
    assert choices['a']['selected']=='dense'
    with patch('ghost_backend.execution.policy.native_fmm_available',return_value=False):
        record['backend_candidates'].pop('dense')
        with pytest.raises(RuntimeError,match='No compatible backend'):
            select_batch_backends([record],1,1,4,validate_options({}))


def test_execution_reservation_survives_nested_profiles_and_restores():
    from ghost_backend.execution.options import allocated_memory_budget
    with patch.object(solver,'_detect_available_gb',return_value=20):
        with execution_scope(dict(ram_budget_gib=12),memory_budget_gib=2):
            assert solver._solve_memory_limit_gb()==2
            with execution_scope(dict(ram_budget_gib=15),memory_budget_gib=8):
                assert solver._solve_memory_limit_gb()==2
        assert allocated_memory_budget() is None


def test_automatic_compressed_rejection_uses_an_admitted_dense_retry():
    from ghost_backend.linalg.hierarchical import HierarchicalRejected
    choice=dict(requested='adaptive',selected='compressed',retry_order=['dense'],reason='test fixture')
    with patch('ghost_backend.execution.selection.select_backend',return_value=choice), \
         patch('ghost_backend.compressed.factor.CompressedFactor.solve',side_effect=HierarchicalRejected('error bound')):
        result=solver.solve_monostatic_rcs_2d(fixture('rectangle',48),[1.],[0.],geometry_units='meters')
    assert result['metadata']['backend_selection']['selected']=='dense'
    assert result['metadata']['backend_selection']['failed_attempts'][0]['backend']=='compressed'
