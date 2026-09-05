"""FREDDY stack -> CSV -> actual 2D and BoR solves of the SAME scalar IBC.

Small analytic bodies test the handoff, not the finite-layer approximation
and not runtime at ten-foot body dimensions.
"""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Backend'))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'FREDDY'))
from ibc.compute import LoadedLayer, MaterialTable, compute_stack_impedance_many
from ibc.io import write_output
import rcs_solver as rcs
import bor_dispatch
from mie_reference import sigma_impedance_cylinder
from mie_sphere import sigma_impedance_sphere
from test_rcs_physics_regression import _circle_segment


class PecStackHandoffTests(unittest.TestCase):
    def test_scalar_ibc_from_stack_reaches_both_solvers_over_1_to_18_ghz(self):
        frequencies=[1.,9.5,18.]
        layer=LoadedLayer(.000762,False,0.,MaterialTable([1.,18.],[4-.1j]*2,[1.]*2),None)
        impedance=compute_stack_impedance_many(frequencies,[layer],'pec')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write_output(root/'coating.csv',[(f,z.real,z.imag) for f,z in zip(frequencies,impedance)],True)
            radius=.006
            circle={'segments':[_circle_segment(radius,64,2,ibc=1)],'ibcs':[['1','coating.csv']],'dielectrics':[]}
            angle=np.linspace(0.,np.pi,33)
            points=np.column_stack((radius*np.sin(angle),radius*np.cos(angle)))
            sphere={'segments':[{'name':'sphere','seg_type':2,'properties':['2','1','1','0','0'],
                     'point_pairs':[dict(x1=a[0],y1=a[1],x2=b[0],y2=b[1]) for a,b in zip(points[:-1],points[1:])]}],
                    'ibcs':circle['ibcs'],'dielectrics':[]}
            options=dict(geometry_units='meters',material_base_dir=str(root))
            two=rcs.solve_monostatic_rcs_2d(circle,frequencies,[0.,45.,90.],**options)
            bor=bor_dispatch.solve_monostatic_rcs_bor(sphere,frequencies,[0.,45.,90.],workers=2,assembly='streaming',**options)
            for index,frequency in enumerate(frequencies):
                for pol,scalar in (('VV','TE'),('HH','TM')):
                    expected=sigma_impedance_cylinder(radius,impedance[index],frequency*1e9,scalar)
                    for row in two['co_solved_samples'][pol]:
                        if row['frequency_ghz']==frequency:
                            self.assertLess(abs(10*np.log10(row['rcs_linear']/expected)),.15)
                expected=sigma_impedance_sphere(radius,frequency*1e9,impedance[index])
                for row in bor['samples']:
                    if row['frequency_ghz']==frequency:
                        self.assertLess(abs(10*np.log10(row['rcs_linear']/expected)),.15)


if __name__=='__main__':
    unittest.main()
