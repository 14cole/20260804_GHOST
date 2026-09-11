"""Qualify actual geometry-to-compressed assembly against saved dense systems."""
from streamed_operator import *
from compressed_system import CompressedSystem
from algebra import qr_basis,project
import cpu_execution as ce
import psutil,threading,weakref,argparse
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'ghost_ram_time_20260910'


def run(frequency,pol,tile,cut):
    mesh,infos,k=prepare('airfoil',pol,frequency=frequency)
    oracle=PreparedOracle(mesh,infos,pol,cut=cut)
    xy=np.empty((oracle.n,2))
    for (mi,side),(offset,count) in oracle.layout['dof_map'].items():
        xy[offset:offset+count]=oracle.xy[oracle.layout['ifaces'][mi]['nodes']]
    peaks=[psutil.Process().memory_info().rss];stop=threading.Event()
    def sample():
        while not stop.wait(.05):peaks.append(psutil.Process().memory_info().rss)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start()
    record=dict(frequency=frequency,pol=pol,tile=tile,cut=cut)
    state=ce.CPUState();state.reuse_operators=False;ops.set_assembly_threads(4)
    try:
        start=time.perf_counter()
        with ce._STATE.override(state):operator=StreamedOperator(oracle,xy,tile=tile)
        record['assembly_seconds']=time.perf_counter()-start
        record['operator']=operator.evidence
        start=time.perf_counter();system=CompressedSystem(operator,xy,tolerance=1e-13)
        record['inverse_seconds']=time.perf_counter()-start
        record['compressed_system']=system.evidence
        total_error=operator.row_error+system.row_error
        # System row_norm is the norm of the tiled operator, prior to its
        # second compression. Subtract first-stage error for a lower bound.
        lower_norm=max(float(np.max(system.row_norm)-np.max(operator.row_error)),0.)
        refs=[weakref.ref(oracle),weakref.ref(operator)]
        oracle=operator=None
        record['construction_inputs_released']=all(ref() is None for ref in refs)
        stop.set();monitor.join()
        record['construction_sampled_rss_bytes']=max(peaks)
        prefix=PRIOR/'systems'/('airfoil-n256-f'+str(frequency))/pol
        b=np.load(str(prefix)+'-b.npy')
        basis,recovery,_=qr_basis(b)
        start=time.perf_counter();x=system.apply(basis,solve=True)@recovery
        # Qualification includes every requested illumination and the error
        # incurred in both compression stages plus any analytically bounded tail.
        residual=system.apply(x)-b
        denominator=np.maximum(lower_norm*np.max(abs(x),axis=0)+np.max(abs(b),axis=0),1e-300)
        bound=(np.max(abs(residual),axis=0)+total_error.max()*np.max(abs(x),axis=0))/denominator
        record['solve_and_bound_seconds']=time.perf_counter()-start
        record['original_residual_bound']=float(bound.max())
        a=np.load(str(prefix)+'-a.npy',mmap_mode='r')
        norm=rcs.matrix_inf_norm(a)
        original=np.zeros(b.shape[1]);matrix_error=0.
        for start in range(0,len(a),64):
            original=np.maximum(original,np.max(abs(a[start:start+64]@x-b[start:start+64]),axis=0))
        den=np.maximum(norm*np.max(abs(x),axis=0)+np.max(abs(b),axis=0),1e-300)
        record['independent_backward']=float(np.max(original/den))
        record['field_error']=difference(project(prefix,x),project(prefix,np.load(str(prefix)+'-x.npy')))
        record['accepted']=bool(record['original_residual_bound']<=1e-12 and
            record['independent_backward']<=1e-12 and record['field_error']<=1e-10)
    except Exception as exc:
        record['rejected']=repr(exc)
    finally:
        stop.set();monitor.join()
    name='streamed-f{}-{}-tile{}-cut{}.json'.format(frequency,pol,tile,cut)
    (HERE/name).write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frequency',type=float,default=1.)
    p.add_argument('--pol',default='both');p.add_argument('--tile',type=int,default=256)
    p.add_argument('--cut',type=float);args=p.parse_args()
    for pol in ('TE','TM') if args.pol=='both' else [args.pol]:run(args.frequency,pol,args.tile,args.cut)
