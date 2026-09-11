"""Public monostatic compressed solves, optionally with genuine mesh certification."""
from pathlib import Path
import sys,os,json,time,argparse,hashlib,threading
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'tools/GHOST/Backend'))
import numpy as np
import rcs_solver as rcs,rcs_operators as ops
p=argparse.ArgumentParser();p.add_argument('--kind',default='airfoil');p.add_argument('--frequency',type=float,default=1.)
p.add_argument('--materials',action='store_true');p.add_argument('--certified',action='store_true')
p.add_argument('--factor',default='compressed');p.add_argument('--angles',type=int,default=361)
p.add_argument('--panels',type=int,default=0);p.add_argument('--repeat',type=int,default=1,help='Run-label index; each invocation executes one fresh run.')
args=p.parse_args();os.environ['GHOST_CPU_FACTORIZATION']=args.factor;os.environ['GHOST_DENSE_BACKEND']='cpu'
ops.set_assembly_threads(4)
inputs=json.loads((HERE.parent/'ghost_final_20260910/inputs.json').read_text())
source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'tools/GHOST/Backend').glob('*.py')}
kinds=[k for k in inputs if k not in ('airfoil','ibc4096')] if args.materials else [args.kind]
records=[]
for kind in kinds:
    value=inputs[kind];freq=.6 if args.materials else args.frequency
    if args.panels:
        for s in value['segments']:s['properties'][1]=str(args.panels)
    start=time.perf_counter();r=dict(kind=kind,frequency=freq,factor=args.factor,certified=args.certified,angles=args.angles)
    r['configuration']=dict(assembly_threads=4,blas_threads=2,inner_assembly_tile=ops._ASSEMBLY_TILE,
        compressed_storage_mib=os.environ.get('GHOST_COMPRESSED_STORAGE_MIB','2048'))
    stopped=threading.Event();phase=['initializing']
    def heartbeat():
        import psutil
        while not stopped.wait(30):
            print(kind,phase[0],'elapsed',round(time.perf_counter()-start,1),'RSS MiB',round(psutil.Process().memory_info().rss/2**20),flush=True)
    monitor=threading.Thread(target=heartbeat,daemon=True);monitor.start()
    seen=set()
    def progress(done,total,message):
        label=':'.join(message.split(':')[:2]) if args.certified else message.split(':')[0]
        if label!=phase[0]:
            phase[0]=label;print(kind,label,flush=True)
        if (done==0 or done==total) and (done,total) not in seen:
            seen.add((done,total));print(kind,done,total,message,flush=True)
    try:
        solve=rcs.solve_monostatic_rcs_2d_certified if args.certified else rcs.solve_monostatic_rcs_2d
        extra={} if args.certified else dict(compute_condition_number=True,strict_quality_gate=True)
        result=solve(value,[freq],np.linspace(0,360,args.angles).tolist(),solver_method='experimental_cpu',
            geometry_units='inches' if kind=='airfoil' else 'meters',material_base_dir=str(ROOT),
            max_panels=100000,progress_callback=progress,**extra)
        r.update(seconds=time.perf_counter()-start,metadata=result['metadata'],
            fields={p:[[v['rcs_amp_real'],v['rcs_amp_imag']] for v in rows] for p,rows in result['co_solved_samples'].items()},
            accepted=bool(result['metadata']['quality_gate']['passed'] and
                (not args.certified or result['metadata']['mesh_convergence_certified'])))
    except Exception as exc:
        import traceback
        r.update(seconds=time.perf_counter()-start,rejected=repr(exc));traceback.print_exc()
    finally:stopped.set();monitor.join()
    records.append(r);print(kind,r['seconds'],r.get('accepted',False),r.get('rejected',''),flush=True)
label='materials' if args.materials else args.kind+'-f'+str(args.frequency)
name=label+'-'+args.factor+('-certified' if args.certified else '-base')+'-a'+str(args.angles)+'-r'+str(args.repeat)+'.json'
(HERE/name).write_text(json.dumps(dict(runs=records,source_hashes=source_hashes),indent=2))
sys.exit(0 if all(r.get('accepted') for r in records) else 1)
