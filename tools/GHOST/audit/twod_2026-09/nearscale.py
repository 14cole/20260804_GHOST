import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
from ghost_backend.twod.pulse import coefficients as C
from test_fmm_efficiency import mesh_for

mesh,k=mesh_for('reentrant',2048,3.);g=AssemblyGeometry(mesh);k=float(np.real(k))
rows=np.arange(32);cols=np.arange(512)
rr=np.repeat(rows,len(cols));cc=np.tile(cols,len(rows))
d=np.linalg.norm(g.centers[rr]-g.centers[cc],axis=1)
near=d<=3*np.maximum(g.lengths[rr],g.lengths[cc])
nr,nc=rr[near],cc[near]
print('tile 32x512 = %d pairs, of which %d near (%.1f%%)'%(len(rr),near.sum(),100*near.mean()))

def timed(fn,threads,repeats=8):
    fn()
    start=time.perf_counter()
    if threads==1:
        for _ in range(repeats):fn()
    else:
        with ThreadPoolExecutor(max_workers=threads) as pool:list(pool.map(lambda _:fn(),range(repeats)))
    return time.perf_counter()-start

cases={
 'accurate_pairs (near only)': lambda: C.accurate_pairs(g,k,nr,nc,{'S','KP'}),
 'near_pairs order 16':        lambda: C.near_pairs(g,k,nr,nc,{'S','KP'},16),
 'near_pairs, no self terms':  lambda: C.near_pairs(g,k,nr[nr!=nc],nc[nr!=nc],{'S','KP'},16),
 'self_single_layer x %d'%(nr==nc).sum(): lambda: [C.self_single_layer(k,float(L)) for L in g.lengths[nc[nr==nc]]],
 'far only (point_pairs)':     lambda: C.point_pairs(g,k,rr[~near],cc[~near],{'S','KP'},6),
}
print('%-30s %8s %8s %8s   %s'%('layer','1 thr','2 thr','4 thr','4-thread speedup'))
for name,fn in cases.items():
    t1=timed(fn,1);t4=timed(fn,4);t2=timed(fn,2)
    print('%-30s %7.3fs %7.3fs %7.3fs   %.2fx'%(name,t1,t2,t4,t1/t4))
