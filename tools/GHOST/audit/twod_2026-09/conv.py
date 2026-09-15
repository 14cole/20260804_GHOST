from common import *
import numpy as np
F=1.0  # GHz
R=.06
k=2*np.pi*F*1e9/s.C0
print('ka=',k*R)
rows=[]
for case in ['pec','ibc','lossy']:
    for disc in ['galerkin','pulse']:
        line=[]
        for n in [64,128,256,512,1024]:
            r=run(fixture(case,n),F,[0.],disc=disc)
            out={}
            for label,pol in [('VV','TE'),('HH','TM')]:
                if case=='pec':
                    ref=pec_cylinder_backscatter_amplitude(R,F*1e9,pol)
                    got=fields(r,label)[0]
                    out[pol]=abs(got-ref)/abs(ref)
                else:
                    ref=(sigma_dielectric_cylinder(R,3-.1j,1.,F*1e9,pol) if case=='lossy'
                         else sigma_impedance_cylinder(R,50-10j,F*1e9,pol))
                    got=sigma(r,label)[0]
                    out[pol]=abs(got-ref)/abs(ref)
            line.append((n,out['TM'],out['TE']))
        print(case,disc)
        for n,tm,te in line:print('   n=%5d  TM %.3e  TE %.3e'%(n,tm,te))
