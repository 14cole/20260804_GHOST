from common import *
import numpy as np
R=.06;ka=3.8317059702075125
f=ka*s.C0/(2*np.pi*R)/1e9
ref=pec_cylinder_backscatter_amplitude(R,f*1e9,'TE')
print('reference TE amplitude',ref,' |ref|',abs(ref))
for disc in ['galerkin','pulse']:
    for cert in [False,True]:
        eo=dict(factorization='dense')
        if disc!='galerkin':eo['discretization']=disc
        fn=s.solve_monostatic_rcs_2d_certified if cert else s.solve_monostatic_rcs_2d
        kw=dict(geometry_units='meters',solver_method='experimental_cpu',execution_options=eo)
        if not cert:kw['compute_condition_number']=True
        try:
            r=fn(fixture('pec',512),[f],[0.],**kw)
            got=fields(r,'VV')[0]
            md=r['metadata']
            print('%-9s certified=%-5s err=%.3e cond=%s gate=%s'%(
                disc,cert,abs(got-ref)/abs(ref),md.get('condition_est_max'),
                (md.get('quality_gate') or {}).get('passed')))
            if (md.get('quality_gate') or {}).get('violations'):
                print('    violations:',md['quality_gate']['violations'])
        except Exception as e:
            print('%-9s certified=%-5s RAISED %s: %s'%(disc,cert,type(e).__name__,str(e)[:300]))
