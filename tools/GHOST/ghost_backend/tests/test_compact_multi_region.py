"""Compact coefficients, phase lifetimes and solver memory contracts."""
import gc
import math
import os
from pathlib import Path
import sys
import unittest
from unittest import mock
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import ghost_backend.twod.solver as rcs
import ghost_backend.twod.operators as ops
import ghost_backend.twod.formulations.regions as multi_region
import ghost_backend.execution.cpu as cpu_execution
from ghost_backend.twod.assembly.compact import CompactOperator
from ghost_backend.linalg.workspace import first_nonfinite, matrix_inf_norm
from ghost_backend.linalg.refined_lu import RefinedLU, linear_precision
from test_experimental_cpu import fixture, solve, fields


def prepared(kind='layered', pol='TM', count=24):
    snapshot = fixture('layered' if kind == 'equal_k' else kind, count)
    if kind == 'equal_k':
        snapshot['dielectrics'] = [['1','3','0','1','0'], ['2','1','0','3','0']]
    materials = rcs.MaterialLibrary.from_entries(snapshot['ibcs'], snapshot['dielectrics'], base_dir='.')
    freq = .6
    k0 = 2*math.pi*freq*1e9/rcs.C0
    panels = rcs._build_panels(snapshot, 1., rcs._mesh_wavelength_for_snapshot(snapshot, materials, freq)[0])
    infos = rcs._build_coupled_panel_info(panels, materials, freq, pol, k0)
    mesh, _ = rcs._build_linear_mesh_interface_aware(panels, infos)
    return mesh, infos, k0


class CompactMultiRegionTests(unittest.TestCase):
    def test_rectangular_outputs_match_full_coefficients_with_tiles_and_masks(self):
        mesh, infos, k0 = prepared()
        n, e = len(mesh.nodes), len(mesh.elements)
        masks = [np.arange(e) < e//2, np.arange(e) >= e//2, np.zeros(e,dtype=bool)]
        rows = [np.arange(0,n,2), np.arange(n//2,n), np.arange(3)]
        columns = [np.arange(n//2), np.arange(n//2,n), np.arange(2)]
        weights = [None, np.linspace(.2,2,e)*(1+.3j), None]
        for derivative in (False, True):
            for compaction in (0., 1.1):
                with self.subTest(derivative=derivative, compaction=compaction), \
                     mock.patch.object(ops,'_ASSEMBLY_TILE',11), \
                     mock.patch.object(ops,'_ASSEMBLY_THREADS',2), \
                     mock.patch.object(ops,'_ASSEMBLY_COMPACT_BELOW',compaction):
                    args = dict(mesh=mesh,k0=k0*(1-.02j),obs_normal_deriv=derivative,
                        source_element_masks=masks,compute_double_layer_many=[True,False,True],
                        single_layer_observation_coefficients_many=weights)
                    full = ops._assemble_linear_operator_matrices_multi(**args)
                    compact = ops._assemble_linear_operator_matrices_multi(
                        output_node_ids_many=list(zip(rows,columns)),**args)
                    for i,(actual,expected) in enumerate(zip(compact,full)):
                        for a,b in zip(actual,expected):
                            np.testing.assert_allclose(a.values,b[np.ix_(rows[i],columns[i])],rtol=1e-13,atol=1e-18)
                            self.assertEqual(a.shape,(len(rows[i]),len(columns[i])))

    def test_full_global_arrays_are_not_allocated_and_operators_die_before_lu(self):
        mesh, infos, k0 = prepared('coated', count=48)
        n = len(mesh.nodes)
        original_zeros = np.zeros
        original_ops = rcs._assemble_linear_operator_matrices_multi
        original_solve = rcs._SCIPY_LINALG.lu_factor
        refs = []
        def zeros(shape,*args,**kwargs):
            if isinstance(shape,tuple) and shape == (n,n):
                self.fail('Allocated a global dense operator/mass matrix')
            return original_zeros(shape,*args,**kwargs)
        def assembly(*args,**kwargs):
            values = original_ops(*args,**kwargs)
            refs.extend(weakref.ref(a.values if hasattr(a, 'values') else a) for pair in values for a in pair)
            return values
        def factor(a,*args,**kwargs):
            gc.collect()
            self.assertTrue(refs)
            self.assertTrue(all(ref() is None for ref in refs))
            return original_solve(a,*args,**kwargs)
        with mock.patch.object(np,'zeros',zeros), \
             mock.patch.object(ops,'_ASSEMBLY_TILE',17), \
             mock.patch.object(rcs,'_assemble_linear_operator_matrices_multi',assembly), \
             mock.patch.object(rcs._SCIPY_LINALG,'lu_factor',factor):
            result = rcs._solve_multi_region_indirect(mesh,infos,'TM',k0,np.array([0.,90.]),return_density=False)
        self.assertIsNone(result[3])
        self.assertTrue(np.all(np.isfinite(result[0])))

    def test_plan_counts_compact_allocations_including_equal_k_and_weighted_operators(self):
        for kind in ('layered','coated','mixed','equal_k'):
            for pol in ('TE','TM'):
                mesh,infos,k0 = prepared(kind,pol)
                layout = multi_region.build_layout(mesh,infos,pol)
                resources = multi_region.storage_resources(mesh,layout)
                actual = []
                original = rcs._assemble_linear_operator_matrices_multi
                def capture(*args,**kwargs):
                    outputs = original(*args,**kwargs)
                    actual.extend(a.values.size for pair in outputs for a in pair if any(a.values.strides))
                    return outputs
                with mock.patch.object(rcs,'_assemble_linear_operator_matrices_multi',capture):
                    matrix,_ = multi_region._assemble_system_fresh(mesh,infos,pol)
                self.assertEqual(sum(actual),resources['operator_entries'])
                self.assertEqual(len(actual),resources['operator_matrices'])
                self.assertEqual(matrix.shape[0],layout['n_dof'])

    def test_equal_wavenumber_row_union_matches_global_storage_system(self):
        for pol in ('TE','TM'):
            mesh,infos,k0 = prepared('equal_k',pol)
            original = rcs._assemble_linear_operator_matrices_multi
            def global_storage(*args,**kwargs):
                kwargs.pop('output_node_ids_many')
                kwargs.pop('double_layer_output_node_ids_many')
                return original(*args,**kwargs)
            with mock.patch.object(rcs,'_assemble_linear_operator_matrices_multi',global_storage):
                reference,_ = multi_region._assemble_system_fresh(mesh,infos,pol)
            compact,_ = multi_region.assemble_system(mesh,infos,pol)
            np.testing.assert_allclose(compact,reference,rtol=2e-12,atol=1e-18)

    def test_memory_gate_uses_phase_maximum_and_same_batch_limit(self):
        mesh,infos,_ = prepared()
        resources = rcs._dense_formulation_resources(mesh,infos,'TM')
        args = dict(nnodes=resources['nodes'],use_cfie=False,system_dofs=resources['system_dofs'],
                    operator_matrices=resources['operator_matrices'],dense_resources=resources,
                    formulation=resources['formulation'],n_rhs=18001)
        one = rcs._estimate_memory_gb(**dict(args,n_rhs=1))
        sweep = rcs._estimate_memory_gb(**args)
        with mock.patch.dict(os.environ,{'GHOST_CPU_ANGLE_BATCH_SIZE':'7'}):
            streamed = rcs._estimate_memory_gb(solver_method='experimental_cpu',**args)
            self.assertEqual(cpu_execution.CPUState().batch_size,7)
        self.assertGreater(sweep,one)
        # Both CPU modes now use the same bounded batches. Kernel-table/cache
        # reservations are additional to the common direct-solve budget.
        self.assertLess(streamed, sweep + (cpu_execution.CACHE_BYTES+cpu_execution.TABLE_BYTES)/1024**3)
        capped = rcs._estimate_memory_gb(**dict(args,n_rhs=256))
        self.assertAlmostEqual(sweep-capped, (18001-256)*4096/1024**3)
        for bad in ('0','-1','257','bad'):
            with mock.patch.dict(os.environ,{'GHOST_CPU_ANGLE_BATCH_SIZE':bad}):
                with self.assertRaises(ValueError):
                    cpu_execution.CPUState()

    def test_small_angle_batches_keep_all_angles_and_one_factorization(self):
        angles = [float(x) for x in range(0,361,10)]
        reference = solve(fixture('layered',24),angles=angles)
        with mock.patch.dict(os.environ,{'GHOST_CPU_ANGLE_BATCH_SIZE':'7'}):
            result = solve(fixture('layered',24),'experimental_cpu',angles)
        for pol in ('VV','HH'):
            np.testing.assert_allclose(fields(result,pol),fields(reference,pol),rtol=1e-10,atol=1e-13)
            self.assertEqual(len(fields(result,pol)),len(angles))
        for event in result['metadata']['experimental_cpu']['systems']:
            self.assertEqual(event['factorizations'],1)
            self.assertEqual(event['max_rhs_columns'],7)
            self.assertEqual(event['rhs_batches'],6)

    def test_checks_handle_fortran_strides_and_nonfinite_indices_in_blocks(self):
        rng = np.random.RandomState(41)
        a = np.array(rng.randn(513,257)+1j*rng.randn(513,257),order='F')
        self.assertAlmostEqual(matrix_inf_norm(a,4096)/np.linalg.norm(a,ord=np.inf),1.,places=14)
        self.assertIsNone(first_nonfinite(a,1024))
        a[512,256] = complex(np.inf,0)
        self.assertEqual(first_nonfinite(a,1024),(512,256))
        with self.assertRaisesRegex(ValueError,r'\(512, 256\)'):
            rcs._ensure_finite_linear_system(a)

    def test_mixed_adjoint_refinement_matches_double_without_matrix_conjugate(self):
        rng = np.random.RandomState(18)
        a = rng.randn(30,30)+1j*rng.randn(30,30)+30*np.eye(30)
        b = rng.randn(30,3)+1j*rng.randn(30,3)
        factor = RefinedLU(a)
        for trans,op in ((0,a),(1,a.T),(2,a.conj().T)):
            np.testing.assert_allclose(factor.solve(b,trans),np.linalg.solve(op,b),rtol=1e-10,atol=1e-12)

    def test_failed_mixed_factor_is_released_before_double_fallback(self):
        refs = []
        class FailedFactor:
            def __init__(self,a):
                self.lu = a.astype(np.complex64)
                refs.append(weakref.ref(self.lu))
            def solve(self,rhs,return_residual=False):
                raise np.linalg.LinAlgError('refinement stalled')
        original = rcs._SCIPY_LINALG.lu_factor
        def factor(*args,**kwargs):
            self.assertTrue(all(ref() is None for ref in refs))
            return original(*args,**kwargs)
        with linear_precision('mixed'), mock.patch.object(rcs,'RefinedLU',FailedFactor), \
             mock.patch.object(rcs._SCIPY_LINALG,'lu_factor',factor):
            actual = rcs._solve_dense_system(np.eye(4,dtype=complex),np.ones((4,2)))
        np.testing.assert_array_equal(actual,np.ones((4,2)))


if __name__ == '__main__':
    unittest.main()
