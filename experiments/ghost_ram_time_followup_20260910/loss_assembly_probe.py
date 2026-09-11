from support import *
from loss_traversal import pruned,STATS
from ownership_probe import amplitudes
from contextlib import nullcontext,contextmanager
from unittest.mock import patch
import inspect
import argparse
import boundary_fields,sweep_compression
import assembly_variants
from owned_assembly import clone

@contextmanager
def original_checks(records):
    original=boundary_fields.solve_fields;signature=inspect.signature(original);count=[0]
    def checking(*args,**kwargs):
        p=signature.bind(*args,**kwargs);p.apply_defaults();values=p.arguments
        pol=('TE','TM')[count[0]];count[0]+=1
        original_a=np.load(PRIOR/'systems/airfoil-n256-f2.0'/(pol+'-a.npy'),mmap_mode='r')
        a=values['matrix'];norm=rcs.matrix_inf_norm(original_a)
        errors=np.zeros(len(a))
        for start in range(0,len(a),64):errors[start:start+64]=np.sum(abs(a[start:start+64]-original_a[start:start+64]),axis=1)
        record=dict(pol=pol,coefficient_inf_error=float(errors.max()/norm),original_backward=0.,residual_bound=0.)
        solve=sweep_compression.solve
        def checking_solve(factor,rhs):
            x=solve(factor,rhs)
            den=np.maximum(norm*np.max(abs(x),axis=0)+np.max(abs(rhs),axis=0),1e-300)
            record['original_backward']=max(record['original_backward'],float(np.max(np.max(abs(original_a@x-rhs),axis=0)/den)))
            record['residual_bound']=max(record['residual_bound'],float(np.max((np.max(abs(a@x-rhs),axis=0)+errors.max()*np.max(abs(x),axis=0))/den)))
            return x
        with patch.object(sweep_compression,'solve',checking_solve):result=original(*args,**kwargs)
        records.append(record);return result
    with patch.object(boundary_fields,'solve_fields',checking):yield

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cut',type=int,default=0);p.add_argument('--repeat',type=int,default=1);p.add_argument('--check',action='store_true');p.add_argument('--variant',default='baseline');args=p.parse_args()
    ops.set_assembly_threads(4);checks=[]
    with patch.object(assembly_variants,'clone',clone),pruned(args.cut) if args.cut else nullcontext(),assembly_variants.variant(args.variant),original_checks(checks) if args.check else nullcontext():
        start=time.perf_counter();result=public_solve('airfoil',frequency=2.0)
    seconds=time.perf_counter()-start
    suffix=('-'+args.variant if args.variant!='baseline' else '')+('-check' if args.check else '')
    write('loss-assembly-cut{}-r{}{}.json'.format(args.cut,args.repeat,suffix),
        dict(seconds=seconds,checks=checks,stats=dict(STATS),metadata=result['metadata'],
            fields={p:[[z.real,z.imag] for z in v] for p,v in amplitudes(result).items()}))
    print(args.cut,args.check,seconds,checks,dict(STATS),flush=True)
