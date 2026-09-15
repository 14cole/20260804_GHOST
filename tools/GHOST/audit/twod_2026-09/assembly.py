import sys,time,cProfile,pstats,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np
from ghost_backend.execution.options import execution_scope
from ghost_backend.twod.pulse.kernel import PulseKernel,PulseSystem
from ghost_backend.twod.pulse.runtime import PulseOracle,dense_matrix
from test_fmm_efficiency import mesh_for

mesh,k=mesh_for('reentrant',2048,3.)
def build():
    f=PulseKernel(mesh,k)
    a=PulseSystem(f.n);a.add(f,'S');a.add(f,'KP')
    a.pulse_weights={'S':np.ones(f.n),'KP':np.ones(f.n)}
    return PulseOracle(a)
oracle=build()
for threads in (1,2,4):
    with execution_scope(dict(assembly_threads=threads)):
        dense_matrix(oracle,lambda:None)
        t=time.perf_counter();m=dense_matrix(oracle,lambda:None);e=time.perf_counter()-t
    print('dense_matrix  threads=%d  %6.3f s'%(threads,e))
with execution_scope(dict(assembly_threads=4)):
    pr=cProfile.Profile();pr.enable();dense_matrix(oracle,lambda:None);pr.disable()
out=io.StringIO();pstats.Stats(pr,stream=out).sort_stats('tottime').print_stats(10)
print('\n'.join(out.getvalue().split('\n')[4:18]))
