import sys,cProfile,pstats,io,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np
import ghost_backend.twod.solver as s
from test_2d_capability_acceptance import _circle

def fixture(n):
    return dict(segments=[_circle('body',.06,n,2)],ibcs=[],dielectrics=[])

def run(n,mode,angles=3):
    return s.solve_monostatic_rcs_2d(fixture(n),[1.0],list(np.linspace(0,60,angles)),
        geometry_units='meters',solver_method='experimental_cpu',
        compute_condition_number=False,
        execution_options=dict(discretization='pulse',factorization=mode,assembly_threads=1))

n=int(sys.argv[1]);mode=sys.argv[2]
run(64,mode,1)                       # warm imports/caches
t=time.perf_counter();pr=cProfile.Profile();pr.enable()
run(n,mode)
pr.disable();elapsed=time.perf_counter()-t
out=io.StringIO();st=pstats.Stats(pr,stream=out).sort_stats('tottime')
st.print_stats(22)
print('n=%d mode=%s  elapsed %.2f s'%(n,mode,elapsed))
print('\n'.join(out.getvalue().split('\n')[4:32]))
