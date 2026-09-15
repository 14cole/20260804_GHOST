from common import *
import numpy as np
from test_2d_capability_acceptance import _segment
F=1.0
def rect(panels,kind='pec'):
    # 0.6 m x 0.25 m PEC rectangle, clockwise
    pts=[(.3,.125),(.3,-.125),(-.3,-.125),(-.3,.125),(.3,.125)]
    if kind=='pec':
        return dict(segments=[_segment('body',2,pts,panels=panels)],ibcs=[],dielectrics=[])
    return dict(segments=[_segment('body',3,pts,panels=panels,pos=1)],ibcs=[],
                dielectrics=[['1','3','-0.1','1','0']])
for kind in ['pec','lossy']:
    ref={}
    r=run(rect(256,kind),F,[30.],disc='galerkin')
    for p in ['VV','HH']:ref[p]=fields(r,p)[0]
    print(kind,'reference panel count',r['metadata'].get('panel_count'))
    for disc in ['galerkin','pulse']:
        for panels in [12,24,48,96]:
            r=run(rect(panels,kind),F,[30.],disc=disc)
            e={p:abs(fields(r,p)[0]-ref[p])/abs(ref[p]) for p in ['VV','HH']}
            print('  %-8s panels/side=%3d n=%4d  TM %.3e  TE %.3e'%(
                disc,panels,r['metadata'].get('panel_count'),e['HH'],e['VV']))
