"""The multi-region pulse route takes a different oracle path; check it too."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np
import ghost_backend.twod.solver as s
from test_experimental_cpu import fixture,fields
for case in ('lossy','coated'):
    out={}
    for threads in (1,4):
        t=time.perf_counter()
        r=s.solve_monostatic_rcs_2d(fixture(case,1024),[1.0],[0.,37.],geometry_units='meters',
            solver_method='experimental_cpu',compute_condition_number=False,
            execution_options=dict(discretization='pulse',factorization='dense',assembly_threads=threads))
        out[threads]=(time.perf_counter()-t,fields(r,'HH'))
    same=np.array_equal(out[1][1],out[4][1])
    print('%-7s 1 thread %6.2f s, 4 threads %6.2f s (%.2fx)  identical fields: %s'%(
        case,out[1][0],out[4][0],out[1][0]/out[4][0],same))
