"""Reproducible qualification of an imported geometry in fresh worker processes.

Normal runs choose Automatic. Explicit backends here are comparison experiments,
not settings required by GUI or HPC users. No job is submitted to a cluster.
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time


def run_worker(command,env,timeout):
    # A Windows virtual-environment launcher can own another Python process.
    # Cancel the worker tree, not just that launcher, on timeout/interruption.
    import psutil
    flags=({'creationflags':subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
           if os.name=='nt' else {'start_new_session':True})
    process=subprocess.Popen(command,env=env,**flags)
    try:
        status=process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired,KeyboardInterrupt):
        if process.poll() is None:
            try:children=psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:children=[]
            try:
                for child in reversed(children):
                    try:child.kill()
                    except psutil.NoSuchProcess:pass
            finally:
                if process.poll() is None:process.kill()
            _,alive=psutil.wait_procs(children,timeout=5)
            if alive:raise RuntimeError('A benchmark descendant did not stop after cancellation.')
        process.wait()
        raise
    if status:raise subprocess.CalledProcessError(status,command)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('geometry',type=Path)
    parser.add_argument('--frequencies',type=float,nargs='+',default=[1.])
    parser.add_argument('--angles',type=int,default=19)
    parser.add_argument('--units',default='inches')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--ram-gib',type=float,default=None)
    parser.add_argument('--storage-mib',type=int,default=2048)
    parser.add_argument('--modes',nargs='+',choices=['auto','dense','compressed','fmm'],default=['auto'])
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--certified',action='store_true')
    parser.add_argument('--mesh-strategy',choices=['adaptive','global','local'],default='adaptive',
                        help='Accuracy strategy for comparison; normal production uses Automatic.')
    parser.add_argument('--condition-estimate',action='store_true',
                        help='Include condition estimation in raw timings; certification always includes it.')
    parser.add_argument('--forecast-only',action='store_true')
    parser.add_argument('--timeout-seconds',type=float,default=1800.)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.angles<1 or args.repeats<1 or args.threads<1 or args.timeout_seconds<=0:
        parser.error('Counts and timeout must be positive.')
    if not args.worker:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        rows=[]
        env=dict(os.environ,OPENBLAS_NUM_THREADS=str(args.threads),OMP_NUM_THREADS=str(args.threads),
                 MKL_NUM_THREADS=str(args.threads))
        for frequency in args.frequencies:
            for mode in args.modes:
                for repeat in range(args.repeats):
                    output=args.output.with_name(args.output.stem+'-{}-{}-{}.json'.format(frequency,mode,repeat))
                    command=[sys.executable,str(Path(__file__).resolve()),str(args.geometry.resolve()),
                        '--worker','--frequencies',str(frequency),'--modes',mode,'--units',args.units,
                        '--angles',str(args.angles),'--threads',str(args.threads),'--storage-mib',str(args.storage_mib),
                        '--output',str(output.resolve()),'--mesh-strategy',args.mesh_strategy]
                    if args.ram_gib is not None:command+=['--ram-gib',str(args.ram_gib)]
                    if args.certified:command+=['--certified']
                    if args.condition_estimate:command+=['--condition-estimate']
                    if args.forecast_only:command+=['--forecast-only']
                    try:
                        run_worker(command,env,args.timeout_seconds)
                        row=json.loads(output.read_text(encoding='utf-8'))
                    except subprocess.TimeoutExpired:
                        row=dict(frequency_ghz=frequency,mode=mode,error='Benchmark worker exceeded the time limit; no result accepted.')
                    rows.append(row)
                    args.output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
        return int(any(row.get('error') for row in rows))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import scipy
    from ghost_backend.geometry.io import parse_geometry,build_geometry_snapshot
    from ghost_backend.twod import solver
    from ghost_backend.execution.options import validate_options
    from ghost_backend.execution.selection import select_backend
    from ghost_backend.execution.provenance import backend_source_records,source_bundle_fingerprint
    from ghost_backend.execution.thread_control import threadpool_info
    raw=args.geometry.read_bytes()
    snapshot=build_geometry_snapshot(*parse_geometry(raw.decode('utf-8-sig')))
    snapshot['source_path']=str(args.geometry.resolve())
    mode=args.modes[0]
    options=validate_options(dict(factorization='adaptive' if mode=='auto' else mode,mesh_strategy=args.mesh_strategy,
        assembly_threads=args.threads,blas_threads=args.threads,compressed_storage_mib=args.storage_mib,
        ram_budget_gib=args.ram_gib))
    arguments=dict(geometry_snapshot=snapshot,frequencies_ghz=args.frequencies,
        elevations_deg=np.linspace(0.,360.,args.angles).tolist(),geometry_units=args.units,
        material_base_dir=str(args.geometry.resolve().parent),max_panels=100000,
        solver_method='auto' if mode=='auto' else 'experimental_cpu')
    def source_identity():
        records=backend_source_records(str(Path(__file__).resolve().parents[1]/'ghost_backend'),
                                       {'benchmark_solver.py':str(Path(__file__).resolve())})
        return source_bundle_fingerprint(records)
    source_fingerprint=source_identity()
    record=dict(geometry_sha256=hashlib.sha256(raw).hexdigest(),frequencies_ghz=args.frequencies,
        angles=args.angles,units=args.units,mode=mode,options=options,certified=args.certified,
        condition_estimate=bool(args.certified or args.condition_estimate),
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
        platform=platform.platform(),processor=platform.processor(),blas=threadpool_info(),
        backend_source_fingerprint=source_fingerprint)
    started=time.perf_counter()
    try:
        if args.forecast_only:record['selection']=select_backend(arguments,options,args.certified)
        else:
            fun=solver.solve_monostatic_rcs_2d_certified if args.certified else solver.solve_monostatic_rcs_2d
            if not args.certified:arguments['compute_condition_number']=args.condition_estimate
            result=fun(**arguments,execution_options=options)
            if source_identity()!=source_fingerprint:
                raise RuntimeError('Solver source changed during the benchmark; no result accepted.')
            record['metadata']=result['metadata']
            record['fields']={pol:[[v['rcs_amp_real'],v['rcs_amp_imag']] for v in rows]
                              for pol,rows in result['co_solved_samples'].items()}
    except (RuntimeError,ValueError,MemoryError,ArithmeticError) as exc:
        record['error']=type(exc).__name__+': '+str(exc)
    record['wall_seconds']=time.perf_counter()-started
    args.output.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(mode,args.frequencies,round(record['wall_seconds'],3),record.get('error','OK'),flush=True)


if __name__=='__main__':raise SystemExit(main())
