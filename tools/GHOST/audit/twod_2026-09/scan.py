from common import *
import numpy as np
R=.06
for ka in np.arange(3.60,4.05,.05):
    f=ka*s.C0/(2*np.pi*R)/1e9
    ref=pec_cylinder_backscatter_amplitude(R,f*1e9,'TE')
    out=[]
    for disc in ['galerkin','pulse']:
        r=run(fixture('pec',512),f,[0.],disc=disc)
        out.append(abs(fields(r,'VV')[0]-ref)/abs(ref))
    print('ka=%.3f  galerkin %.3e   pulse %.3e'%(ka,out[0],out[1]))
