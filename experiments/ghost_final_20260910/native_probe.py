"""Native material algebra: accurate tiles plus coarse checked preconditioning."""
from preconditioned_sweep import *
from checked_compressed_system import Oracle
from algebra import project
HERE=Path(__file__).resolve().parent;PRIOR=HERE.parent/'ghost_ram_time_20260910'
class Exact(Oracle):
    dropped_routes=0
    def get_with_error(self,rows,cols):
        value=self.get(rows,cols);return value,np.zeros(value.shape)
records=[]
for kind in json.loads((HERE/'inputs.json').read_text()):
    if kind in ('airfoil','ibc4096'):continue
    for pol in ('TE','TM'):
        prefix=PRIOR/'systems'/(kind+'-n384-f0.6')/pol
        record=dict(kind=kind,pol=pol)
        if not Path(str(prefix)+'-a.npy').exists():record['skipped']='Zero contrast';records.append(record);continue
        try:
            a=np.load(str(prefix)+'-a.npy',mmap_mode='r');b=np.load(str(prefix)+'-b.npy');xy=np.load(str(prefix)+'-xy.npy')
            start=time.perf_counter();op=StreamedOperator(Exact(a),xy,tile=128)
            solver=PreconditionedSweep(op,xy)
            record['build_seconds']=time.perf_counter()-start
            start=time.perf_counter();x=solver.solve(b);record['solve_seconds']=time.perf_counter()-start
            record['evidence']=solver.evidence;record['dense_pair_bytes']=2*a.nbytes
            den=np.maximum(rcs.matrix_inf_norm(a)*np.max(abs(x),axis=0)+np.max(abs(b),axis=0),1e-300)
            record['backward']=float(np.max(np.max(abs(a@x-b),axis=0)/den))
            record['field_error']=difference(project(prefix,x),project(prefix,np.load(str(prefix)+'-x.npy')))
            record['accepted']=bool(record['backward']<=1e-12 and record['field_error']<=1e-10)
        except Exception as exc:record['rejected']=repr(exc)
        records.append(record);print(json.dumps(record),flush=True)
        a=b=xy=op=solver=x=None
(HERE/'native-results.json').write_text(json.dumps(records,indent=2))
sys.exit(0 if sum(r.get('accepted',False) for r in records)==26 and
         sum('skipped' in r for r in records)==2 else 1)
