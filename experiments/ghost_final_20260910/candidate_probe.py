"""Isolate inverse/preconditioner choices using saved native matrices."""
from preconditioned_sweep import *
from checked_compressed_system import Oracle
from algebra import project
import argparse
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'ghost_ram_time_20260910'
p=argparse.ArgumentParser();p.add_argument('--kind',default='airfoil');p.add_argument('--frequency',type=float,default=2.)
p.add_argument('--tolerances',default='1e-4,1e-6,1e-8');p.add_argument('--pol',default='TE');args=p.parse_args()
prefix=PRIOR/'systems'/('{}-n{}-f{}'.format(args.kind,256 if args.kind=='airfoil' else 384,args.frequency))/args.pol
a=np.load(str(prefix)+'-a.npy',mmap_mode='r');b=np.load(str(prefix)+'-b.npy');xy=np.load(str(prefix)+'-xy.npy')
class Exact(Oracle):
    dropped_routes=0
    def get_with_error(self,rows,cols):
        value=self.get(rows,cols);return value,np.zeros(value.shape)
start=time.perf_counter();operator=StreamedOperator(Exact(a),xy,tile=512)
records=[]
for tolerance in map(float,args.tolerances.split(',')):
    record=dict(kind=args.kind,pol=args.pol,tolerance=tolerance,operator_bytes=operator.bytes)
    try:
        start=time.perf_counter();solver=PreconditionedSweep(operator,xy,tolerance=tolerance)
        record['factor_seconds']=time.perf_counter()-start
        start=time.perf_counter();x=solver.solve(b);record['solve_seconds']=time.perf_counter()-start
        record['evidence']=solver.evidence
        den=np.maximum(rcs.matrix_inf_norm(a)*np.max(abs(x),axis=0)+np.max(abs(b),axis=0),1e-300)
        record['backward']=float(np.max(np.max(abs(a@x-b),axis=0)/den))
        record['field_error']=difference(project(prefix,x),project(prefix,np.load(str(prefix)+'-x.npy')))
        record['accepted']=bool(record['backward']<=1e-12 and record['field_error']<=1e-10)
    except Exception as exc:record['rejected']=repr(exc)
    records.append(record);print(json.dumps(record),flush=True)
    solver=x=None
(HERE/('candidate-{}-{}.json'.format(args.kind,args.pol))).write_text(json.dumps(records,indent=2))
