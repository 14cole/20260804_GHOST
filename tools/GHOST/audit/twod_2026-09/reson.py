from common import *
import numpy as np
R=.06
# interior Dirichlet eigenvalues (J0,J1 zeros) and Neumann (J0',J1' zeros)
cases=[('TM/Dirichlet ka=2.4048',2.404825557695773),
       ('TM/Dirichlet ka=5.5201',5.520078110286311),
       ('TM/Dirichlet ka=3.8317',3.831705970207512),
       ('TE/Neumann  ka=1.8412',1.8411837813406593),
       ('TE/Neumann  ka=3.0542',3.0542369282271403),
       ('TE/Neumann  ka=3.8317',3.8317059702075125)]
for name,ka in cases:
    f=ka*s.C0/(2*np.pi*R)/1e9
    pol='TM' if name.startswith('TM') else 'TE'
    label='HH' if pol=='TM' else 'VV'
    ref=pec_cylinder_backscatter_amplitude(R,f*1e9,pol)
    row=[]
    for disc,mode in [('galerkin','dense'),('galerkin','fmm'),('pulse','dense'),('pulse','fmm')]:
        try:
            r=run(fixture('pec',512),f,[0.],disc=disc,mode=mode)
            got=fields(r,label)[0]
            row.append('%s/%s %.2e'%(disc[:3],mode[:3],abs(got-ref)/abs(ref)))
        except Exception as e:
            row.append('%s/%s ERR %s'%(disc[:3],mode[:3],type(e).__name__))
    print('%-24s'%name,'  '.join(row))
