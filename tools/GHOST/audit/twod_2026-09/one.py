import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np
import ghost_backend.twod.solver as s
from test_2d_capability_acceptance import _circle
def rss():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmHWM'):return int(line.split()[1])/1024
mode,n,threads=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
t=time.perf_counter()
r=s.solve_monostatic_rcs_2d(dict(segments=[_circle('body',.06,n,2)],ibcs=[],dielectrics=[]),
    [1.0],list(np.linspace(0,60,3)),geometry_units='meters',solver_method='experimental_cpu',
    compute_condition_number=False,
    execution_options=(dict(factorization='dense',assembly_threads=threads) if mode=='galerkin' else dict(discretization='pulse',factorization=mode,assembly_threads=threads)))
print('%-8s n=%-6d threads=%-3d %7.2f s  peak %7.1f MiB  (dense matrix %.1f MiB)'%(
    mode,n,threads,time.perf_counter()-t,rss(),16*n*n/1024**2 if mode=='dense' else 0.))
