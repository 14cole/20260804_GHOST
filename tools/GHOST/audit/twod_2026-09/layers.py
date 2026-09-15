"""Which layer stops scaling: raw ufuncs, green(), point_pairs(), blocks()?"""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.special import y0,j0
from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
from ghost_backend.twod.pulse import coefficients as C
from test_fmm_efficiency import mesh_for

mesh,k=mesh_for('reentrant',2048,3.);g=AssemblyGeometry(mesh);k=float(np.real(k))
rows=np.arange(0,2048);cols=np.arange(0,2048)
z=np.abs(np.random.default_rng(0).normal(size=(4096,6)))+.5

def timed(fn,threads,repeats=8):
    fn()
    start=time.perf_counter()
    if threads==1:
        for _ in range(repeats):fn()
    else:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(lambda _:fn(),range(repeats)))
    return time.perf_counter()-start

cases={
 'scipy y0+j0 ufuncs only': lambda: (y0(z),j0(z)),
 'green()':                 lambda: C.green(k,z,True,True),
 'point_pairs(4096 pairs)': lambda: C.point_pairs(g,k,rows[:4096%2048 or 2048],cols[:2048],{'S','KP'},6),
 'blocks(32x512 tile)':     lambda: C.blocks(g,k,rows[:32],cols[:512],{'S','KP'},6,True),
}
print('%-26s %8s %8s %8s   %s'%('layer','1 thr','2 thr','4 thr','4-thread speedup'))
for name,fn in cases.items():
    t1=timed(fn,1);t2=timed(fn,2);t4=timed(fn,4)
    print('%-26s %7.3fs %7.3fs %7.3fs   %.2fx'%(name,t1,t2,t4,t1/t4))
