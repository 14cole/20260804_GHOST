"""Actual paired geometry -> accurate tiles -> coarse inverse -> checked sweep."""
from paired_operators import *
from preconditioned_sweep import PreconditionedSweep
from cpu_kernels import farfield
from algebra import project
import cpu_execution as ce
import argparse,threading,psutil,weakref
HERE=Path(__file__).resolve().parent;PRIOR=HERE.parent/'ghost_ram_time_20260910'
p=argparse.ArgumentParser();p.add_argument('--frequency',type=float,default=2.)
p.add_argument('--tile',type=int,default=512);p.add_argument('--tolerance',type=float,default=1e-6)
p.add_argument('--budget-mib',type=float,default=512);p.add_argument('--cut',type=float,default=32)
p.add_argument('--spool',action='store_true')
p.add_argument('--repeat',type=int,default=1)
args=p.parse_args();budget=int(args.budget_mib*1024**2)
peaks=[];stop=threading.Event()
def sample():
    while not stop.wait(.05):peaks.append(psutil.Process().memory_info().rss)
monitor=threading.Thread(target=sample,daemon=True);monitor.start()
record=dict(frequency=args.frequency,tile=args.tile,tolerance=args.tolerance,budget=budget,spooled=args.spool,channels={})
pair=[]
try:
    start=time.perf_counter()
    mesh,te,k=prepare('airfoil','TE',frequency=args.frequency)
    _,tm,_=prepare('airfoil','TM',frequency=args.frequency)
    oracle=PairedOracle(mesh,te,tm,cut=args.cut);layouts=[o.layout for o in oracle.oracles]
    xy=mr.dof_coordinates(mesh,layouts[0]);angles=np.arange(361.)
    record['preparation_seconds']=time.perf_counter()-start
    state=ce.CPUState();state.reuse_operators=False;ops.set_assembly_threads(4)
    start=time.perf_counter()
    with ce._STATE.override(state):pair=build_pair(oracle,xy,tile=args.tile,budget=budget,
        checkpoint=state.checkpoint,spool_directory=HERE if args.spool else None)
    record['paired_assembly_seconds']=time.perf_counter()-start
    record['shared_kernel_groups']=oracle.kernel_groups
    record['operator_bytes']=sum(o.bytes for o in pair)
    ref=weakref.ref(oracle);oracle=None;record['oracle_released']=ref() is None
    for index,pol in enumerate(('TE','TM')):
        op=pair[index];other=pair[1-index]
        item=dict(operator=op.evidence)
        start=time.perf_counter()
        if isinstance(op,SpooledOperator):op.load()
        item['load_seconds']=time.perf_counter()-start
        start=time.perf_counter();solver=PreconditionedSweep(op,xy,tolerance=args.tolerance,
            budget=budget-(other.bytes if other is not None else 0),checkpoint=state.checkpoint)
        item['factor_seconds']=time.perf_counter()-start
        start=time.perf_counter();b=mr.rhs_many(mesh,layouts[index],k,angles)
        x=solver.solve(b)
        mask,density=mr.exterior_projection(mesh,layouts[index])
        amp=farfield(mesh,density(x),k,angles,'SLP',element_mask=mask)
        item['rhs_solve_projection_seconds']=time.perf_counter()-start
        item['solver']=solver.evidence;item['field']=[[z.real,z.imag] for z in amp]
        # Only qualification needs stored full currents. Disk I/O is outside core time.
        np.save(HERE/('paired-f{}-{}-solution.npy'.format(args.frequency,pol)),x)
        ref=weakref.ref(op);pair[index]=op=solver=other=b=x=density=None
        item['operator_released']=ref() is None
        record['channels'][pol]=item
    stop.set();monitor.join();record['sampled_core_rss_bytes']=max(peaks)
    for pol,item in record['channels'].items():
        prefix=PRIOR/'systems'/('airfoil-n256-f'+str(args.frequency))/pol
        a=np.load(str(prefix)+'-a.npy',mmap_mode='r');b=np.load(str(prefix)+'-b.npy')
        x=np.load(HERE/('paired-f{}-{}-solution.npy'.format(args.frequency,pol)))
        ri=np.zeros(361)
        for row in range(0,len(a),64):ri=np.maximum(ri,np.max(abs(a[row:row+64]@x-b[row:row+64]),axis=0))
        den=np.maximum(rcs.matrix_inf_norm(a)*np.max(abs(x),axis=0)+np.max(abs(b),axis=0),1e-300)
        item['independent_backward']=float(np.max(ri/den))
        amp=np.asarray(item.pop('field'));amp=amp[:,0]+1j*amp[:,1]
        item['field_error']=difference(amp,project(prefix,np.load(str(prefix)+'-x.npy')))
    record['accepted']=all(v['independent_backward']<=1e-12 and v['field_error']<=1e-10
                           and v['operator_released'] for v in record['channels'].values())
except Exception as exc:record['rejected']=repr(exc)
finally:
    stop.set();monitor.join()
    for op in pair:
        if isinstance(op,SpooledOperator):op.close()
(HERE/('paired-f{}-tile{}-tol{}-{}-r{}.json'.format(args.frequency,args.tile,args.tolerance,'spool' if args.spool else 'ram',args.repeat))).write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
sys.exit(0 if record.get('accepted') else 1)
