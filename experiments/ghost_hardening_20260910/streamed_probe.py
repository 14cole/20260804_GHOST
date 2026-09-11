"""Qualify actual geometry-to-compressed assembly against saved dense systems."""
from streamed_operator import *
from checked_compressed_system import CompressedSystem
from algebra import qr_basis,project
import cpu_execution as ce
import psutil,threading,weakref,argparse
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'ghost_ram_time_20260910'


def run(frequency,pol,tile,cut,compression='qr',budget=512*1024**2):
    mesh,infos,k=prepare('airfoil',pol,frequency=frequency)
    oracle=PreparedOracle(mesh,infos,pol,cut=cut)
    xy=np.empty((oracle.n,2))
    for (mi,side),(offset,count) in oracle.layout['dof_map'].items():
        xy[offset:offset+count]=oracle.xy[oracle.layout['ifaces'][mi]['nodes']]
    peaks=[psutil.Process().memory_info().rss];stop=threading.Event()
    def sample():
        while not stop.wait(.05):peaks.append(psutil.Process().memory_info().rss)
    monitor=threading.Thread(target=sample,daemon=True);monitor.start()
    record=dict(frequency=frequency,pol=pol,tile=tile,cut=cut,compression=compression,combined_payload_budget=budget)
    state=ce.CPUState();state.reuse_operators=False;ops.set_assembly_threads(4)
    try:
        start=time.perf_counter()
        with ce._STATE.override(state):operator=StreamedOperator(oracle,xy,tile=tile,compression=compression,budget=budget,checkpoint=state.checkpoint)
        record['assembly_seconds']=time.perf_counter()-start
        record['operator']=operator.evidence
        start=time.perf_counter();system=CompressedSystem(operator,xy,tolerance=1e-13,budget=budget-operator.bytes,checkpoint=state.checkpoint)
        record['inverse_seconds']=time.perf_counter()-start
        record['compressed_system']=system.evidence
        record['combined_retained_bytes']=operator.bytes+system.bytes
        input_error=operator.row_error
        refs=[weakref.ref(oracle),weakref.ref(operator)]
        oracle=operator=None
        record['construction_inputs_released']=all(ref() is None for ref in refs)
        record['construction_sampled_rss_bytes']=max(peaks)
        prefix=PRIOR/'systems'/('airfoil-n256-f'+str(frequency))/pol
        b=np.load(str(prefix)+'-b.npy')
        basis,recovery,_=qr_basis(b)
        start=time.perf_counter();x=system.apply(basis,solve=True)@recovery
        x=system.solve_checked(b,input_error=input_error,initial=x,overwrite_initial=True)
        # Qualification includes every requested illumination and the error
        # incurred in both compression stages plus any analytically bounded tail.
        record['solve_and_bound_seconds']=time.perf_counter()-start
        record['original_residual_bound']=system.evidence['last_original_residual_bound']
        stop.set();monitor.join()
        record['assembly_inverse_rhs_and_check_sampled_rss_bytes']=max(peaks)
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
    name='streamed-f{}-{}-tile{}-cut{}-{}.json'.format(frequency,pol,tile,cut,compression)
    (HERE/name).write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--frequency',type=float,default=1.)
    p.add_argument('--pol',default='both');p.add_argument('--tile',type=int,default=256)
    p.add_argument('--cut',type=float)
    p.add_argument('--compression',choices=('qr','svd'),default='qr')
    p.add_argument('--budget-mib',type=float,default=512)
    args=p.parse_args()
    records=[run(args.frequency,pol,args.tile,args.cut,args.compression,int(args.budget_mib*1024**2))
             for pol in (('TE','TM') if args.pol=='both' else [args.pol])]
    sys.exit(0 if all(r.get('accepted',False) for r in records) else 1)
