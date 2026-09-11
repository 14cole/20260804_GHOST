from pathlib import Path
import sys,os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'ghost_ram_time_20260910'))
from common import *
from compressed_native import NativeOracle
from assembly_geometry import AssemblyGeometry
import robin_system,dielectric_system,sheet_system,thin_sheet,boundary_fields
import unittest
from unittest.mock import patch


class NativeQueryTests(unittest.TestCase):
    def test_regional_pairing_and_pruning_bounds(self):
        import multi_region as mr
        from compressed_oracle import PairedOracle
        dropped=0
        for kind in ('pec_ibc','coated','layered','mixed'):
            value=snapshot(kind,20)
            for material in value['dielectrics']:material[2]='-400'
            if value['dielectrics']:
                for segment in value['segments']:segment['properties'][1]='4'
            mesh,te,k=prepare(kind,'TE',value=value)
            _,tm,_=prepare(kind,'TM',value=value)
            oracle=PairedOracle(mesh,te,tm,cut=2.)
            references=[mr._assemble_system_fresh(mesh,infos,pol)[0]
                        for infos,pol in ((te,'TE'),(tm,'TM'))]
            order=np.arange(oracle.n)
            for c0 in range(0,oracle.n,13):
                cols=order[c0:c0+13]
                for r0 in range(0,oracle.n,17):
                    rows=order[r0:r0+17]
                    for (actual,bound),reference in zip(oracle.get_with_error(rows,cols),references):
                        expected=reference[np.ix_(rows,cols)]
                        slack=5e-13*max(float(np.max(abs(reference))),1e-300)
                        self.assertTrue(np.all(abs(actual-expected)<=bound+slack))
            dropped+=sum(o.dropped_routes for o in oracle.oracles)
        self.assertGreater(dropped,0)

    def test_hypersingular_subsets_masks_and_order(self):
        mesh,infos,k=prepare('mixed_sheet','TE',count=18)
        n=len(mesh.nodes);rng=np.random.RandomState(4)
        for orders in ((4,4),(4,8)):
            for mask in (None,np.arange(len(mesh.elements))%2==0):
                expected=rcs._assemble_linear_hypersingular_matrix(mesh,k,obs_order=orders[0],src_order=orders[1],source_element_mask=mask)
                rr=rng.permutation(n)[:n//2];cc=rng.permutation(n)[:n//3]
                for rows,cols in ((rr,cc),(cc,rr),(rr,rr),([],cc)):
                    actual=rcs._assemble_linear_hypersingular_matrix(mesh,k,obs_order=orders[0],src_order=orders[1],
                        source_element_mask=mask,output_node_ids=(rows,cols),prepared_geometry=AssemblyGeometry(mesh))
                    np.testing.assert_allclose(actual.values,expected[np.ix_(np.asarray(rows,int),cols)],rtol=2e-12,atol=3e-14)

    def test_native_formulations_against_original_assembly(self):
        for kind in ('pec','ibc','pec_ibc','lossless','lossy','magnetic','sheet','mixed_sheet','thin','thin_magnetic'):
            for pol in ('TE','TM'):
                with self.subTest(kind=kind,pol=pol):
                    mesh,infos,k=prepare(kind,pol,count=40)
                    layer=None
                    if kind.startswith('thin'):
                        value=snapshot(kind,40);materials=rcs.MaterialLibrary.from_entries(value['ibcs'],value['dielectrics'],str(ROOT))
                        layer=thin_sheet.layer_for_mesh(mesh,materials,.6);saved=[]
                        def capture(mesh,matrix,k,angles,*a,**kw):
                            saved.append(matrix.copy());return np.zeros(len(angles)),np.zeros(len(angles)),0.,None
                        with patch.object(boundary_fields,'solve_fields',capture):
                            thin_sheet.solve_thin_layer_fields(mesh,k,[0.],pol,*layer)
                        expected=saved[0];mesh=thin_sheet._continuous_oriented_mesh(mesh);form='thin'
                    elif kind in ('pec','ibc','pec_ibc'):
                        expected,_,_=robin_system.assemble_system(mesh,infos,pol,k);form='robin'
                    elif kind in ('sheet','mixed_sheet'):
                        expected,_=sheet_system.assemble_system(mesh,infos,pol,k);form='sheet'
                    else:expected=dielectric_system.assemble_system(mesh,infos,pol,k);form='dielectric'
                    oracle=NativeOracle(mesh,infos,pol,k,form,layer=layer)
                    rng=np.random.RandomState(65);order=rng.permutation(oracle.n)
                    actual=np.zeros_like(expected)
                    for start in range(0,oracle.n,19):
                        cols=order[start:start+19];oracle.prepare_columns(cols)
                        for r0 in range(0,oracle.n,23):
                            rows=order[r0:r0+23];block,error=oracle.get_with_error(rows,cols)
                            actual[np.ix_(rows,cols)]=block
                    error=np.max(abs(actual-expected))/max(np.max(abs(expected)),1e-300)
                    self.assertLess(error,3e-12)
                    self.assertEqual(oracle.entries,oracle.n**2)

if __name__=='__main__':unittest.main()
