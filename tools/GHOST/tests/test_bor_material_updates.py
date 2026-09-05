"""Independent radial references for transmitting sheets and reactive IBC."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Backend'))
from bor_solver import solve_bor
from bor_kernels import C0
from mie_sphere import sigma_electric_sheet_sphere, sigma_impedance_sphere, sigma_pec_sphere
import bor_dispatch


def sphere(ka,n):
    r=ka*C0/(2*np.pi*1e9)
    a=np.linspace(0,np.pi,n+1)
    return r,np.column_stack((r*np.sin(a),r*np.cos(a)))


class BorMaterialUpdateTests(unittest.TestCase):
    def test_sheet_reference_limits(self):
        r,_=sphere(1.5,30)
        self.assertAlmostEqual(sigma_electric_sheet_sphere(r,1e9,0),sigma_pec_sphere(r,1e9),places=12)
        self.assertLess(sigma_electric_sheet_sphere(r,1e9,1e12),1e-15)

    def test_transmitting_sheet_sphere_matches_two_sided_boundary_reference(self):
        r,points=sphere(1.5,30); z=100+20j
        result=solve_bor(points,1e9,[0.,45.,90.],sheet_zs=z,assembly='streaming',workers=2)
        truth=sigma_electric_sheet_sphere(r,1e9,z)
        for key in ('sigma_vv','sigma_hh'):
            self.assertLess(float(np.max(abs(10*np.log10(np.asarray(result[key])/truth)))),.15)
        self.assertEqual(result['boundary_model'],'transmitting_electric_sheet')

    def test_reactive_ibc_at_interior_resonance_matches_mie(self):
        r,points=sphere(4.4934,45); z=100j
        result=solve_bor(points,1e9,[0.,45.,90.],formulation='cfie',zs=z,assembly='streaming',workers=2)
        truth=sigma_impedance_sphere(r,1e9,z)
        for key in ('sigma_vv','sigma_hh'):
            self.assertLess(float(np.max(abs(10*np.log10(np.asarray(result[key])/truth)))),.2)

    def test_dispatch_selects_cfie_only_for_uniform_reactive_ibc(self):
        self.assertEqual(bor_dispatch._conductor_formulation([100j,100j]),'cfie')
        self.assertEqual(bor_dispatch._conductor_formulation([100j,120j]),'efie')
        self.assertEqual(bor_dispatch._conductor_formulation([50+10j,50+10j]),'efie')

    def test_unsupported_boundary_combinations_fail_before_assembly(self):
        _, points = sphere(.8, 12)
        with self.assertRaisesRegex(ValueError, 'uniform Zs'):
            solve_bor(points, 1e9, [0.], formulation='cfie', zs=np.linspace(100, 120, 12)*1j)
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            solve_bor(points, 1e9, [0.], sheet_zs=100, zs=100j)
        from rcs_solver import MaterialLibrary
        library = MaterialLibrary.from_entries([['1','thin_dielectric','.001','2']],
                                               [['2','3','0','1','0']], '.')
        with self.assertRaisesRegex(ValueError, 'thin'):
            library.get_impedance(1, 1.)

    def test_public_sheet_route_and_preview_use_the_sheet_formulation(self):
        r,points=sphere(.8,18)
        snapshot={'segments':[{'name':'film','seg_type':1,'properties':['1','1','1','0','0'],
                   'point_pairs':[dict(x1=a[0],y1=a[1],x2=b[0],y2=b[1]) for a,b in zip(points[:-1],points[1:])]}],
                  'ibcs':[['1','constant','100','20','0','0']],'dielectrics':[]}
        preview=bor_dispatch.estimate_bor_resources(snapshot,1.,[0.,90.],geometry_units='meters',mesh_certification=False)
        self.assertEqual(preview['geometry_kind'],'sheet')
        output=bor_dispatch.solve_monostatic_rcs_bor(snapshot,[1.],[0.,90.],geometry_units='meters',assembly='streaming',workers=2)
        truth=sigma_electric_sheet_sphere(r,1e9,100+20j)
        for row in output['samples']:
            self.assertLess(abs(10*np.log10(row['rcs_linear']/truth)),.3)
        self.assertIn('transmitting electric sheet',output['metadata']['formulation'])


if __name__=='__main__':
    unittest.main()
