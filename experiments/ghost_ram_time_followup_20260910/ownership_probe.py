from support import *
from owned_assembly import owned
from contextlib import nullcontext
import argparse

def amplitudes(result):
    return {pol:np.asarray([complex(v['rcs_amp_real'],v['rcs_amp_imag']) for v in rows])
        for pol,rows in result['co_solved_samples'].items()}

def qualification():
    from material_probe import KINDS
    records=[]
    for kind in KINDS:
        reference=json.loads((PRIOR/'systems'/('{}-n384-f0.6'.format(kind))/'public-result.json').read_text())
        with owned():result=public_solve(kind,384)
        old,new=amplitudes(reference),amplitudes(result)
        records.append(dict(kind=kind,field_error={p:difference(new[p],old[p]) for p in old},
            backward=[r['max_backward_error'] for r in result['metadata'].get('experimental_cpu',{}).get('systems',[])],
            reuse=result['metadata'].get('assembled_system_reuses'),gate=result['metadata']['quality_gate']['passed']))
        print(kind,records[-1],flush=True)
    write('ownership-materials.json',records)

def benchmark(args):
    ops.set_assembly_threads(args.threads)
    with owned() if args.mode=='owned' else nullcontext():
        start=time.perf_counter();result=public_solve(args.kind,args.count,angles=361)
    write('ownership-{}-{}-n{}-t{}-r{}.json'.format(args.kind,args.mode,args.count,args.threads,args.repeat),
        dict(seconds=time.perf_counter()-start,metadata=result['metadata'],
            fields={p:[[z.real,z.imag] for z in v] for p,v in amplitudes(result).items()}))
    print(args.kind,args.mode,time.perf_counter()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind');p.add_argument('--mode',default='baseline');p.add_argument('--count',type=int,default=4096)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--repeat',type=int,default=1);args=p.parse_args()
    if args.kind:benchmark(args)
    else:qualification()
