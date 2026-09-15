import sys,time,cProfile,pstats,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np
import ghost_backend.twod.solver as s
from test_2d_capability_acceptance import _circle
def run(n,threads,pr=None):
    if pr:pr.enable()
    r=s.solve_monostatic_rcs_2d(dict(segments=[_circle('body',.06,n,2)],ibcs=[],dielectrics=[]),
        [1.0],list(np.linspace(0,60,3)),geometry_units='meters',solver_method='experimental_cpu',
        compute_condition_number=False,
        execution_options=dict(factorization='dense',assembly_threads=threads,blas_threads=threads))
    if pr:pr.disable()
    return r
run(256,1)
for threads in (1,4):
    t=time.perf_counter();run(2048,threads);print('galerkin dense n=2048 threads=%d  %6.2f s'%(threads,time.perf_counter()-t))
pr=cProfile.Profile();run(2048,1,pr)
out=io.StringIO();pstats.Stats(pr,stream=out).sort_stats('tottime').print_stats(12)
print('\n'.join(out.getvalue().split('\n')[4:20]))
