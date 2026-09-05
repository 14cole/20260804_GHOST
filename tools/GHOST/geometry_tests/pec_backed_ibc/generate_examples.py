"""Create a new folder of illustrative FREDDY-to-GHOST coating artifacts."""
from pathlib import Path
import argparse
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools/FREDDY'))
sys.path.insert(0,str(ROOT/'tools/GHOST/Backend'))
from ibc.compute import LoadedLayer, MaterialTable, compute_stack_impedance_many
from ibc.io import write_output, write_material_table
from ibc.ghost_coating import assess_scalar_coating
from geometry_io import snapshot_to_geometry_text


def generate(folder):
    # New folders only: never replace an edited example or a user's materials.
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    frequencies=[1.+.1*i for i in range(171)]
    material=MaterialTable([1.,18.],[4-.1j]*2,[1.]*2)
    stack=[LoadedLayer(.03*.0254,False,0.,material,None)]
    z=compute_stack_impedance_many(frequencies,stack,'pec')
    write_output(folder/'example_coating_30mil.csv',[(f,v.real,v.imag) for f,v in zip(frequencies,z)],True)
    write_material_table(folder/'example_dielectric.csv',material,True)
    report=assess_scalar_coating(frequencies,stack)
    report['example_material']='Illustrative epsilon 4-j0.1, mu 1; not user material or a validated RCS coating.'
    (folder/'planar_assessment.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for label,count,stop in (('2d',128,-2*math.pi),('bor',64,math.pi)):
        radius=2.03  # inches: 2-inch PEC core + 0.03-inch coating
        angles=[stop*i/count for i in range(count+1)]
        points=[(radius*math.cos(a),radius*math.sin(a)) if label=='2d'
                else (max(0.,radius*math.sin(a)),radius*math.cos(a)) for a in angles]
        snapshot=dict(title='Illustrative_PEC_stack_IBC_'+label+'_units_inches',
                      segments=[dict(name='outer_coating_envelope',seg_type=2,properties=['2','0','10','0','0'],
                                 point_pairs=[dict(x1=a[0],y1=a[1],x2=b[0],y2=b[1]) for a,b in zip(points[:-1],points[1:])])],
                      ibcs=[['10','example_coating_30mil.csv']],dielectrics=[])
        (folder/(label+'_outer_envelope.geo')).write_text(snapshot_to_geometry_text(snapshot),encoding='utf-8')
    return folder


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',help='New output directory (must not already exist)')
    print(generate(parser.parse_args().output))
