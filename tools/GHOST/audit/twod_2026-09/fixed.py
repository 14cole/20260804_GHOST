"""Fixed polygon, refined panels: isolates discretization error from geometry error."""
from common import *
import numpy as np
from test_2d_capability_acceptance import _segment

R=.06;SIDES=24;F=1.0
def poly(kind,panels):
    th=np.linspace(0,-2*np.pi,SIDES+1)
    pts=[(R*np.cos(t),R*np.sin(t)) for t in th]
    if kind=='pec':
        return dict(segments=[_segment('body',2,pts,panels=panels)],ibcs=[],dielectrics=[])
    if kind=='ibc':
        return dict(segments=[_segment('body',2,pts,panels=panels,ibc=1)],
                    ibcs=[['1','constant','50','-10','50','-10']],dielectrics=[])
    return dict(segments=[_segment('body',3,pts,panels=panels,pos=1)],ibcs=[],
                dielectrics=[['1','3','-0.1','1','0']])

for kind in ['pec','ibc','lossy']:
    ref={}
    r=run(poly(kind,64),F,[0.],disc='galerkin')
    for p in ['VV','HH']:ref[p]=fields(r,p)[0]
    print(kind,'reference panels',r['metadata'].get('panel_count',r['metadata'].get('n_panels','?')))
    for disc in ['galerkin','pulse']:
        for panels in [1,2,4,8,16]:
            r=run(poly(kind,panels),F,[0.],disc=disc)
            e={p:abs(fields(r,p)[0]-ref[p])/abs(ref[p]) for p in ['VV','HH']}
            print('  %-8s panels/side=%3d  n=%5d  TM %.3e  TE %.3e'%(
                disc,panels,SIDES*panels,e['HH'],e['VV']))
