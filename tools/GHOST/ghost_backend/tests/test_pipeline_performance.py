"""Accuracy, lifecycle, and resource-choice contracts for the 2-D pipeline."""
import copy
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND.parent))
from ghost_backend.twod import solver
from ghost_backend.twod.preparation import prepare_geometry, preparation_scope
from ghost_backend.execution.options import execution_scope, validate_options, current_options
from ghost_backend.execution.selection import select_backend
from ghost_backend.geometry.spatial import overlapping_pairs
from ghost_backend.twod.meshing import segment_wavelengths
from ghost_backend.twod.checkpoints import FrequencyCheckpoints, run_checkpointed, input_identity
from ghost_backend.compressed.operator import tile_payload
from test_experimental_cpu import fixture, fields


def args(snapshot=None):
    return dict(geometry_snapshot=snapshot or fixture('pec', 48), frequencies_ghz=[.6, .8],
                elevations_deg=[0., 90., 180.], geometry_units='meters', solver_method='experimental_cpu',
                max_panels=10000)


def small_result(frequency, certified=True):
    rows = [dict(frequency_ghz=frequency, theta_inc_deg=0., theta_scat_deg=0., polarization=pol,
                 rcs_amp_real=1., rcs_amp_imag=-.25, rcs_db=-1.5) for pol in ('VV', 'HH')]
    return dict(samples=rows, co_solved_samples={p: [r for r in rows if r['polarization']==p] for p in ('VV','HH')},
                metadata=dict(mesh_convergence_certified=certified, panel_count_min=10, warnings=[]))


class PipelinePerformanceTests(unittest.TestCase):
    def test_spatial_candidates_match_brute_force_including_tolerance(self):
        rng = np.random.RandomState(64)
        for transpose in (False, True):
            xy = rng.rand(200, 4)
            bounds = [(min(a,b),max(a,b),min(c,d),max(c,d)) for a,b,c,d in xy]
            if transpose:
                bounds = [(c,d,a,b) for a,b,c,d in bounds]
            tol = .001
            expected = {(i,j) for i,a in enumerate(bounds) for j,b in enumerate(bounds) if i<j
                        and a[0]<=b[1]+tol and b[0]<=a[1]+tol and a[2]<=b[3]+tol and b[2]<=a[3]+tol}
            actual = list(overlapping_pairs(bounds,tol))
            self.assertEqual(set(actual),expected)
            self.assertEqual(len(actual),len(expected))

    def test_sparse_candidates_and_cooperative_cancellation(self):
        checks = []
        bounds = [(i*2.,i*2.+1,0.,1.) for i in range(10000)]
        self.assertEqual(list(overlapping_pairs(bounds, checkpoint=lambda: checks.append(1))), [])
        self.assertEqual(len(checks),len(bounds))
        parallel = [(0., 100000., float(i), float(i)) for i in range(10000)]
        checks.clear()
        self.assertEqual(list(overlapping_pairs(parallel, checkpoint=lambda: checks.append(1))), [])
        self.assertEqual(len(checks),len(parallel))
        def abort():
            raise InterruptedError('cancel')
        with self.assertRaises(InterruptedError):
            list(overlapping_pairs(bounds,checkpoint=abort))

    def test_preparation_is_shared_per_run_but_invalidated_between_runs(self):
        snapshot = fixture('pec', 24)
        with mock.patch.object(solver.MaterialLibrary, 'from_entries', wraps=solver.MaterialLibrary.from_entries) as load:
            with preparation_scope():
                first = prepare_geometry(snapshot, units='meters')
                self.assertIs(first,prepare_geometry(copy.deepcopy(snapshot), units='meters'))
                changed = copy.deepcopy(snapshot)
                changed['segments'][0]['properties'][1] = '2'
                self.assertIs(first[2],prepare_geometry(changed,units='meters')[2])
                self.assertEqual(load.call_count,1)
            prepare_geometry(snapshot, units='meters')
            self.assertEqual(load.call_count,2)

    def test_certified_sweep_loads_one_material_library(self):
        with mock.patch.object(solver.MaterialLibrary,'from_entries',wraps=solver.MaterialLibrary.from_entries) as load:
            result = solver.solve_monostatic_rcs_2d_certified(**args(), execution_options={})
        self.assertTrue(result['metadata']['mesh_convergence_certified'])
        self.assertEqual(load.call_count,1)
        self.assertEqual(len(result['samples']),12)

    def test_backend_selection_forecasts_fine_mesh_and_preserves_explicit_choice(self):
        with preparation_scope(), mock.patch.object(solver,'_solve_memory_limit_gb',return_value=10):
            plan = select_backend(args(), validate_options(dict(factorization='adaptive')), certified=True)
        self.assertEqual(plan['selected'],'dense')
        self.assertEqual(len(plan['meshes']),8)
        self.assertGreater(max(r['unknowns'] for r in plan['meshes'] if r['phase']=='fine'),
                           max(r['unknowns'] for r in plan['meshes'] if r['phase']=='base'))
        with preparation_scope(), mock.patch.object(solver,'_solve_memory_limit_gb',return_value=.001):
            plan = select_backend(args(), validate_options(dict(factorization='adaptive')), certified=True)
        self.assertEqual(plan['selected'],'compressed')
        with mock.patch('ghost_backend.execution.selection.select_backend',side_effect=AssertionError('manual changed')):
            solver.solve_monostatic_rcs_2d_survey(**dict(args(),frequencies_ghz=[.6]), execution_options=dict(factorization='dense'))

    def test_adaptive_fields_match_dense_and_restore_settings(self):
        kw = dict(args(),frequencies_ghz=[.6])
        direct = solver.solve_monostatic_rcs_2d_certified(**kw,execution_options={})
        adaptive = solver.solve_monostatic_rcs_2d_certified(**kw,execution_options=dict(factorization='adaptive'))
        self.assertEqual(adaptive['metadata']['backend_selection']['selected'],'dense')
        for pol in ('VV','HH'):
            np.testing.assert_allclose(fields(adaptive,pol),fields(direct,pol),rtol=1e-11,atol=1e-13)
        self.assertIsNone(current_options())
        with mock.patch.dict(os.environ, {'GHOST_CPU_FACTORIZATION':'adaptive'}):
            captured=solver.solve_monostatic_rcs_2d_survey(**kw)
            self.assertEqual(captured['metadata']['backend_selection']['selected'],'dense')
            self.assertIsNone(current_options())

    def test_sampled_tile_basis_verifies_full_matrix_and_skips_useless_q(self):
        rng = np.random.RandomState(13)
        raw = (rng.randn(192,5)+1j*rng.randn(192,5)) @ (rng.randn(5,192)+1j*rng.randn(5,192))
        from ghost_backend.linalg.sweep import _qr_basis
        with mock.patch('ghost_backend.compressed.operator._qr_basis',wraps=_qr_basis) as qr:
            payload, recovered, tail, accepted = tile_payload(raw,np.zeros_like(raw.real),1e-14,'qr')
        self.assertTrue(accepted)
        self.assertEqual(qr.call_count,1)
        self.assertEqual(qr.call_args[0][0].shape[1],16)
        np.testing.assert_allclose(recovered,raw,rtol=2e-12,atol=2e-12)
        # A feature entirely outside the sampled columns must still be retained.
        raw[:,7] += rng.randn(192)
        payload,recovered,tail,accepted=tile_payload(raw,np.zeros_like(raw.real),1e-14,'qr')
        self.assertLess(np.linalg.norm(raw-recovered)/np.linalg.norm(raw),1e-14)
        identity=np.eye(192,dtype=complex)
        payload,recovered,tail,accepted=tile_payload(identity,np.zeros((192,192)),1e-14,'qr')
        self.assertFalse(accepted)
        self.assertIs(payload[0],identity)

    def test_local_material_sizing_reduces_only_unprotected_remote_segments(self):
        from test_2d_capability_acceptance import _circle
        snapshot=dict(segments=[_circle('dielectric',.01,64,3,pos=1),
            _circle('remote PEC',.2,64,2,center=(2.,0.))],ibcs=[],dielectrics=[['1','25','-.1','1','0']])
        for segment in snapshot['segments']:
            segment['properties'][1]='0'
        _,_,materials,scale=prepare_geometry(snapshot,units='meters')
        wavelength=solver._mesh_wavelength_for_snapshot(snapshot,materials,.6)[0]
        with execution_scope(dict(mesh_strategy='local')):
            sizes=segment_wavelengths(snapshot,materials,[.6],scale,wavelength)
        self.assertEqual(sizes[0],wavelength)
        self.assertGreater(sizes[1],wavelength)
        global_panels=solver._build_panels(snapshot,scale,wavelength)
        local_panels=solver._build_panels(snapshot,scale,wavelength,segment_wavelengths=sizes)
        self.assertLess(len(local_panels),len(global_panels))
        # Nearby material coupling keeps both boundaries at the global scale.
        for pair in snapshot['segments'][1]['point_pairs']:
            pair['x1']-=1.6;pair['x2']-=1.6
        with execution_scope(dict(mesh_strategy='local')):
            sizes=segment_wavelengths(snapshot,materials,[.6],scale,wavelength)
        self.assertEqual(sizes,[wavelength,wavelength])

    def test_failed_local_certificate_retries_global_without_leaking_scope(self):
        def attempt(*args,**kwargs):
            if current_options()['mesh_strategy']=='local':
                raise ValueError('Certified 2-D mesh convergence failed: changed field')
            return {'metadata':{}}
        with execution_scope(dict(mesh_strategy='local')):
            with mock.patch.object(solver,'_run_certified_2d_pair_impl',side_effect=attempt):
                result=solver._run_certified_2d_pair()
            self.assertEqual(current_options()['mesh_strategy'],'local')
        self.assertEqual(result['metadata']['mesh_strategy_used'],'global')

    def test_checkpoint_roundtrip_corruption_and_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            store=FrequencyCheckpoints(directory,'identity',True)
            result=small_result(.6)
            store.save(.6,result)
            self.assertEqual(store.load(.6),result)
            store._path(.6).write_bytes(b'interrupted output')
            self.assertIsNone(store.load(.6))
            with self.assertRaisesRegex(ValueError,'uncertified'):
                store.save(.6,small_result(.6,False))

    def test_material_file_change_invalidates_checkpoint_and_captured_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'medium.csv'
            path.write_text('frequency_hz,eps_real,eps_imag,mu_real,mu_imag\n600000000,3,-0.1,1,0\n800000000,4,-0.1,1,0\n')
            snapshot=fixture('lossy',24)
            snapshot['dielectrics']=[['1','medium.csv']]
            kw=dict(args(snapshot),material_base_dir=directory)
            with preparation_scope():
                prepare_geometry(snapshot,directory,'meters')
                before=input_identity(kw,{},'double',True)
                path.write_text('frequency_hz,eps_real,eps_imag,mu_real,mu_imag\n600000000,5,-0.1,1,0\n800000000,6,-0.1,1,0\n')
                with self.assertRaisesRegex(ValueError,'changed after run preparation'):
                    input_identity(kw,{},'double',True)
            after=input_identity(kw,{},'double',True)
            self.assertNotEqual(before,after)

    def test_hpc_planner_supports_both_adaptive_routes(self):
        from ghost_backend.geometry.io import Segment, build_geometry_text
        from ghost_backend.hpc.scheduler import predict_2d_resources_many
        snapshot=fixture('pec',24)
        seg=snapshot['segments'][0]
        x=[p[k] for p in seg['point_pairs'] for k in ('x1','x2')]
        y=[p[k] for p in seg['point_pairs'] for k in ('y1','y2')]
        text=build_geometry_text('Test',[Segment('PEC','2',seg['properties'],x,y)],[],[])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'case.geo';path.write_text(text)
            for budget,expected in ((10.,'dense'),(.001,'compressed')):
                with execution_scope(dict(factorization='adaptive')):
                    with mock.patch.object(solver,'_solve_memory_limit_gb',return_value=budget):
                        plan=predict_2d_resources_many(str(path),[.6],['VV','HH'],'meters',10000,
                            fine_factor=1.5,n_angles=3,solver_method='experimental_cpu')
                self.assertEqual({r['backend_selection']['selected'] for r in plan.values()},{expected})
                self.assertTrue(all(r['peak_gb']>0 and r['fine_nodes']>r['nodes'] for r in plan.values()))

    def test_cancelled_sweep_reuses_only_completed_matching_frequencies(self):
        event=threading.Event()
        calls=[]
        def solve(**kw):
            freq=kw['frequencies_ghz'][0]
            calls.append(freq)
            if freq==.6:
                event.set()
            return small_result(freq)
        kw=dict(args(),abort_event=event)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InterruptedError):
                run_checkpointed(solve,kw,directory,validate_options({}),'double',True)
            event.clear()
            result=run_checkpointed(solve,kw,directory,validate_options({}),'double',True)
            self.assertEqual(calls,[.6,.8])
            self.assertEqual(result['metadata']['frequency_checkpoints']['reused'],1)
            self.assertEqual(len(result['samples']),4)
            changed=dict(kw,elevations_deg=[17.])
            self.assertNotEqual(input_identity(kw,{},'double',True),input_identity(changed,{},'double',True))

    def test_frequency_merge_reports_mixed_backend_and_mesh_decisions(self):
        first,second=small_result(.6),small_result(.8)
        for result,backend,strategy in ((first,'dense','local'),(second,'compressed','global')):
            result['metadata'].update(backend_selection=dict(selected=backend),mesh_strategy_used=strategy,
                requested_execution_options=validate_options(dict(factorization='adaptive')),
                execution_options=validate_options(dict(factorization=backend)))
        result=solver._merge_frequency_results(iter([first,second]),[.6,.8])
        self.assertEqual(result['metadata']['backend_selection']['selected'],'mixed')
        self.assertEqual(result['metadata']['execution_options']['factorization'],'adaptive')
        self.assertIn('mixed',result['metadata']['mesh_strategy_used'])


if __name__=='__main__':
    unittest.main()
