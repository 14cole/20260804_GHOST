from common import *
import argparse
from assembly_variants import variant,SCATTER_STATS

def main():
    p=argparse.ArgumentParser();p.add_argument('kind');p.add_argument('--variant',default='baseline')
    p.add_argument('--count',type=int,default=384);p.add_argument('--frequency',type=float,default=.6)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--tile',type=int,default=0)
    p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
    ops.set_assembly_threads(a.threads);ops._ASSEMBLY_TILE=a.tile
    runs=[]
    for repeat in range(a.repeats):
        with variant(a.variant):
            start=time.perf_counter();result=public_solve(a.kind,a.count,a.frequency)
        runs.append(dict(seconds=time.perf_counter()-start,profile=result['metadata']['runtime_profile'],
            field=[[v.real,v.imag] for v in fields(result)],
            fields_by_pol={pol:[[v['rcs_amp_real'],v['rcs_amp_imag']] for v in rows]
                for pol,rows in result['co_solved_samples'].items()},
            residual=result['metadata']['residual_norm_max'],condition=result['metadata']['condition_est_max'],
            gate=result['metadata']['quality_gate']['passed'],scatter=dict(SCATTER_STATS)))
        print(a.kind,a.variant,a.threads,a.tile,round(runs[-1]['seconds'],3),flush=True)
    label='assembly-{}-n{}-f{}-{}-t{}-tile{}'.format(a.kind,a.count,a.frequency,a.variant,a.threads,a.tile)
    write(label+'.json',dict(args=vars(a),runs=runs,median_seconds=float(np.median([r['seconds'] for r in runs]))))

if __name__=='__main__':main()
