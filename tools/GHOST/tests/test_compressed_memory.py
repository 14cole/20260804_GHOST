"""Storage sampling, allocation phases and memory admission regressions."""
import os, sys, unittest, tempfile, threading
from pathlib import Path
from unittest import mock
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(key,'2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Backend'))
import numpy as np
import ghost_backend.twod.solver as rcs
import ghost_backend.compressed.memory as memory
from ghost_backend.compressed.operator import StreamedOperator
from ghost_backend.compressed.inverse import CompressedSystem
from test_experimental_cpu import fixture


class LowRankOracle:
    def __init__(self,n):self.n=n;self.entries=self.calls=self.max_entries=self.dropped_routes=0
    def get_with_error(self,rows,cols):
        self.calls+=1;self.entries+=len(rows)*len(cols)
        self.max_entries=max(self.max_entries,len(rows)*len(cols))
        raw=(1+rows[:,None]/self.n)*(1j+cols[None,:]/self.n)
        raw+=20*(rows[:,None]==cols[None,:])
        return raw,np.zeros(raw.shape)


class CompressedMemoryTests(unittest.TestCase):
    def test_samples_match_known_compressed_storage_without_full_matrix_queries(self):
        oracle=LowRankOracle(1537);xy=np.arange(oracle.n)[:,None]
        sampled=memory.sample_operator(oracle,xy,tile=128)
        self.assertLessEqual(oracle.max_entries,128**2)
        self.assertLessEqual(oracle.calls,6*5)
        actual=StreamedOperator(LowRankOracle(oracle.n),xy,tile=128,budget=64*memory.MIB)
        self.assertLess(abs(sampled['operator_bytes']-actual.bytes)/actual.bytes,.03)
        self.assertGreaterEqual(sampled['operator_allowance_bytes'],actual.bytes*.99)
        second=memory.sample_operator(LowRankOracle(oracle.n),xy,tile=128)
        self.assertEqual(sampled,second)

    def test_inverse_ceiling_covers_real_tree_storage(self):
        n=1025;op=StreamedOperator(LowRankOracle(n),np.arange(n)[:,None],tile=128,budget=64*memory.MIB)
        factor=CompressedSystem(op,op.coordinates,leaf=128,tolerance=1e-6,inverse_only=True,budget=64*memory.MIB)
        ceiling,construction=memory.inverse_storage(n)
        self.assertLessEqual(factor.bytes,ceiling)
        self.assertGreaterEqual(construction,ceiling)

    def plan(self,budget=8*memory.GIB,count=361,safety=1.):
        sample=dict(method='test_sample',operator_bytes=memory.GIB,
            operator_allowance_bytes=int(1.15*memory.GIB),samples=42,sampled=True)
        return memory.forecast(28144,42686,count,min(256,count),4,budget,
            {'compressed_storage':sample},safety=safety)

    def test_unused_capacity_and_spooled_payload_are_not_resident_ram(self):
        first=self.plan();second=self.plan(budget=16*memory.GIB)
        self.assertEqual(first['peak_bytes'],second['peak_bytes'])
        self.assertNotEqual(first['storage_limit_bytes'],second['storage_limit_bytes'])
        self.assertGreater(first['temporary_disk_bytes'],0)
        self.assertEqual(first['peak_bytes'],max(first['phase_bytes'].values()))
        self.assertLess(first['peak_bytes'],8*memory.GIB)

    def test_single_angle_uses_one_rhs_column_and_sweep_batch_is_bounded(self):
        one=self.plan(count=1);sweep=self.plan();long=self.plan(count=10000)
        self.assertEqual(one['inverse_ceiling_bytes'],sweep['inverse_ceiling_bytes'])
        self.assertEqual((sweep['rhs_workspace_bytes']-32*memory.MIB),
            256*(one['rhs_workspace_bytes']-32*memory.MIB))
        self.assertEqual(sweep['rhs_workspace_bytes'],long['rhs_workspace_bytes'])

    def test_safety_multiplier_only_changes_uncertain_operator_storage(self):
        a=self.plan(safety=1.);b=self.plan(safety=2.)
        self.assertEqual(a['inverse_ceiling_bytes'],b['inverse_ceiling_bytes'])
        self.assertEqual(a['rhs_workspace_bytes'],b['rhs_workspace_bytes'])
        self.assertEqual(b['peak_bytes']-a['peak_bytes'],
            b['operator_allowance_bytes']-a['operator_allowance_bytes'])
        for invalid in (0.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):self.plan(safety=invalid)

    def test_explicit_large_kernel_tiles_are_accounted_for(self):
        with mock.patch('ghost_backend.twod.operators._ASSEMBLY_TILE',0):a=self.plan()
        with mock.patch('ghost_backend.twod.operators._ASSEMBLY_TILE',2048):b=self.plan()
        self.assertGreater(b['assembly_workspace_bytes'],a['assembly_workspace_bytes'])
        self.assertGreater(b['phase_bytes']['assembly'],a['phase_bytes']['assembly'])

    def test_thin_strip_tile_metadata_is_not_mistaken_for_disk_payload(self):
        normal=memory.forecast(20000,40000,1,1,1,8*memory.GIB,{'formulation':'single_dielectric'})
        thin=memory.forecast(20000,40000,1,1,1,8*memory.GIB,{'formulation':'thin_dielectric_layer'})
        self.assertGreater(thin['tile_metadata_bytes'],normal['tile_metadata_bytes'])
        self.assertGreater(thin['process_geometry_bytes'],normal['process_geometry_bytes'])

    def test_sampling_cancellation_and_metadata_cache_have_no_retained_operators(self):
        def cancel():raise InterruptedError('canceled')
        with self.assertRaises(InterruptedError):
            memory.sample_operator(LowRankOracle(1100),np.arange(1100)[:,None],checkpoint=cancel)
        from ghost_backend.twod.assembly.session import shared_assembly, current_session
        saved=[]
        @shared_assembly
        def run():
            session=current_session();saved.append(session)
            session.memory_storage[b'example']=dict(operator_bytes=10)
            return {}
        run()
        self.assertFalse(saved[0].memory_storage)

    def test_invalid_sample_rejects_before_unchecked_lapack(self):
        oracle=LowRankOracle(1025)
        def invalid(rows,cols):
            return np.full((len(rows),len(cols)),np.nan,dtype=complex),np.zeros((len(rows),len(cols)))
        oracle.get_with_error=invalid
        with mock.patch('ghost_backend.compressed.operator.tile_payload',side_effect=AssertionError('QR on invalid input')):
            with self.assertRaisesRegex(ValueError,'Invalid coefficient tile'):
                memory.sample_operator(oracle,np.arange(oracle.n)[:,None])

    def test_solver_admission_does_not_reject_only_because_payload_cap_is_large(self):
        with mock.patch.dict(os.environ,{'GHOST_CPU_FACTORIZATION':'compressed',
             'GHOST_DENSE_BACKEND':'cpu','GHOST_COMPRESSED_STORAGE_MIB':'8192'}), \
             mock.patch.object(rcs,'_solve_memory_limit_gb',return_value=1.):
            result=rcs.solve_monostatic_rcs_2d_certified(fixture('pec',48),[.6],[0.,90.],
                solver_method='experimental_cpu',geometry_units='meters')
        self.assertTrue(result['metadata']['quality_gate']['passed'])
        estimates=result['metadata']['experimental_cpu']['memory_estimates']
        self.assertTrue(estimates)
        self.assertTrue(all(p['peak_bytes']<memory.GIB for p in estimates))

    def test_scheduler_exposes_disk_capacity_and_ram_separately(self):
        import ghost_backend.hpc.scheduler as hpc_scheduler
        geometry='''Title: memory planner
Segment: square 2
properties: 2 12 0 0 0
-0.02 -0.02 -0.02 0.02
-0.02 0.02 0.02 0.02
0.02 0.02 0.02 -0.02
0.02 -0.02 -0.02 -0.02
IBCS_Resistances:
Dielectrics:
'''
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ,
             {'GHOST_CPU_FACTORIZATION':'compressed','GHOST_DENSE_BACKEND':'cpu'}):
            path=Path(directory)/'small.geo';path.write_text(geometry)
            plan=hpc_scheduler.predict_2d_resources(str(path),.6,'VV','meters',50000,
                fine_factor=1.5,n_angles=361,solver_method='experimental_cpu')
        self.assertIn('memory_estimate',plan)
        self.assertAlmostEqual(plan['peak_gb'],.6,places=8)
        self.assertGreater(plan['memory_estimate']['temporary_disk_bytes'],0)
        self.assertFalse(plan['memory_estimate']['sampled'])


if __name__=='__main__':unittest.main(verbosity=2)
