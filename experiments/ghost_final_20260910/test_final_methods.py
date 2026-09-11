from paired_operators import *
from preconditioned_sweep import PreconditionedSweep
from checked_compressed_system import Oracle
import unittest,weakref,gc
from unittest.mock import patch


class Exact(Oracle):
    dropped_routes=0
    def get_with_error(self,rows,cols):
        value=self.get(rows,cols);return value,np.zeros(value.shape)


class FinalMethodsTests(unittest.TestCase):
    def test_spool_releases_payload_and_removes_file(self):
        mesh,te,k=prepare('mixed','TE',count=16);_,tm,_=prepare('mixed','TM',count=16)
        oracle=PairedOracle(mesh,te,tm);xy=mr.dof_coordinates(mesh,oracle.oracles[0].layout)
        here=Path(__file__).resolve().parent
        pair=build_pair(oracle,xy,tile=32,spool_directory=here)
        disk=pair[1];path=disk.path
        self.assertTrue(path.exists())
        self.assertTrue(all(v is None for v in disk.tiles.values()))
        with self.assertRaises(ValueError):disk.matmul(np.ones(disk.n))
        with self.assertRaises(ValueError):disk.get([0],[0])
        disk.load();self.assertFalse(path.exists())
        expected,_=mr._assemble_system_fresh(mesh,tm,'TM')
        np.testing.assert_allclose(disk.matmul(np.ones(disk.n)),expected@np.ones(disk.n),rtol=3e-12,atol=3e-15)
        with self.assertRaises(MemoryError):build_pair(oracle,xy,tile=32,spool_directory=here,budget=1000)
        self.assertFalse(list(here.glob('ghost-tm-*.bin')))

    def test_spool_truncation_and_cancellation(self):
        mesh,te,k=prepare('mixed','TE',count=16);_,tm,_=prepare('mixed','TM',count=16)
        oracle=PairedOracle(mesh,te,tm);xy=mr.dof_coordinates(mesh,oracle.oracles[0].layout)
        here=Path(__file__).resolve().parent
        for failure in ('truncate','cancel'):
            pair=build_pair(oracle,xy,tile=32,spool_directory=here);disk=pair[1];path=disk.path
            try:
                if failure=='truncate':
                    disk.file.truncate(0)
                    with self.assertRaises(IOError):disk.load()
                else:
                    def canceled():raise InterruptedError('canceled')
                    disk.checkpoint=canceled
                    with self.assertRaises(InterruptedError):disk.load()
                with self.assertRaises(ValueError):disk.matmul(np.ones(disk.n))
            finally:disk.close()
            self.assertFalse(path.exists())
            with self.assertRaises(ValueError):disk.load()

    def test_paired_material_assembly_matches_independent_systems(self):
        rng=np.random.RandomState(137)
        for kind in ('pec','ibc','pec_ibc','layered','coated','mixed','equal_k'):
            mesh,te,k=prepare(kind,'TE',count=20)
            _,tm,_=prepare(kind,'TM',count=20)
            oracle=PairedOracle(mesh,te,tm,cut=32)
            xy=mr.dof_coordinates(mesh,oracle.oracles[0].layout)
            pair=build_pair(oracle,xy,tile=32)
            self.assertEqual(oracle.calls,len(pair[0].tiles))
            for op,infos,pol in zip(pair,(te,tm),('TE','TM')):
                expected,_=mr._assemble_system_fresh(mesh,infos,pol)
                b=rng.randn(op.n,3)+1j*rng.randn(op.n,3)
                np.testing.assert_allclose(op.matmul(b),expected@b,rtol=4e-12,atol=3e-15)
                self.assertEqual(op.evidence['geometry_coefficients'],op.n**2)
            with self.assertRaises(MemoryError):build_pair(oracle,xy,tile=32,budget=1)

    def test_inverse_only_drops_operator_and_checks_accurate_reference(self):
        rng=np.random.RandomState(92);n=260
        a=np.eye(n)*20+rng.randn(n,5)@(rng.randn(5,n)+1j*rng.randn(5,n))/n
        xy=np.arange(n)[:,None];source=Exact(a)
        operator=StreamedOperator(source,xy,tile=64)
        solver=PreconditionedSweep(operator,xy,tolerance=1e-4)
        with self.assertRaises(ValueError):solver.factor.apply(np.ones(n))
        with self.assertRaises(ValueError):solver.factor.solve_checked(np.ones(n))
        b=rng.randn(n,3)@(rng.randn(3,45)+1j*rng.randn(3,45));b[:,4]=0
        np.testing.assert_allclose(solver.solve(b),np.linalg.solve(a,b),rtol=3e-12,atol=3e-14)
        self.assertLess(solver.factor.bytes,a.nbytes)
        self.assertLessEqual(solver.evidence['original_residual_bound'],1e-12)
        refs=[weakref.ref(source),weakref.ref(a)]
        del source,a
        self.assertTrue(all(v() is None for v in refs))
        operator.row_error.fill(100)
        rejected=PreconditionedSweep(operator,xy)
        with self.assertRaisesRegex(RuntimeError,'acceptance'):rejected.solve(b)

    def test_gmres_repair_budget_and_cancel(self):
        n=16;xy=np.arange(n)[:,None];a=np.diag(np.linspace(1,4,n)).astype(complex)
        operator=StreamedOperator(Exact(a),xy,tile=8)
        solver=PreconditionedSweep(operator,xy)
        b=np.arange(1,n+1).astype(complex)
        with patch.object(solver.factor,'apply',side_effect=lambda x,**kw:x.copy()):
            np.testing.assert_allclose(solver.solve(b),np.linalg.solve(a,b),rtol=3e-12,atol=1e-13)
        self.assertGreater(solver.evidence['gmres_columns'],0)
        with self.assertRaises(MemoryError):PreconditionedSweep(operator,xy,budget=1)
        def canceled():raise InterruptedError('canceled')
        with self.assertRaises(InterruptedError):PreconditionedSweep(operator,xy,checkpoint=canceled)
        for bad in (np.ones((n,0)),np.ones(n-1),np.full(n,np.nan)):
            with self.assertRaises(ValueError):solver.solve(bad)

if __name__=='__main__':unittest.main()
