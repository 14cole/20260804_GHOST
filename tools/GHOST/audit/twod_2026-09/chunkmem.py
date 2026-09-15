"""Scaling and peak memory vs chunk size, and that results are unchanged."""
import sys,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np,tracemalloc
from concurrent.futures import ThreadPoolExecutor
from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
from ghost_backend.twod.pulse import coefficients as C
from test_fmm_efficiency import mesh_for

mesh,k=mesh_for('reentrant',2048,3.);g=AssemblyGeometry(mesh);k=complex(k)
n=len(g.lengths);kinds={'S','KP'};order=8   # order 8 = the conservative rule

def blocks_param(rows,cols,chunk):
    rr=np.repeat(rows,len(cols));cc=np.tile(cols,len(rows))
    res={kind:np.empty(len(rr),complex) for kind in kinds}
    for s in range(0,len(rr),chunk):
        e=min(len(rr),s+chunk);r=rr[s:e];c=cc[s:e]
        d=np.linalg.norm(g.centers[r]-g.centers[c],axis=1)
        near=d<=3*np.maximum(g.lengths[r],g.lengths[c])
        orders=np.full(len(r),order)
        ratio=d/g.lengths[c];el=abs(k)*g.lengths[c]
        orders[(ratio>=6)&(el<=.6)]=min(order,4)
        orders[(ratio>=16)&(el<=.15)]=min(order,3)
        orders[(ratio>=200)&(el<=.01)]=min(order,2)
        for q in np.unique(orders[~near]):
            take=(orders==q)&~near
            part=C.point_pairs(g,k,r[take],c[take],kinds,int(q))
            for kind in kinds:res[kind][s:e][take]=part[kind]
        if np.any(near):
            ex=C.accurate_pairs(g,k,r[near],c[near],kinds)
            for kind in kinds:res[kind][s:e][near]=ex[kind]
    return {kind:v.reshape(len(rows),len(cols)) for kind,v in res.items()}

def assemble(chunk,threads,rowsize=128,colsize=2048):
    out={kind:np.empty((n,n),complex) for kind in kinds}
    def row_block(start):
        rows=np.arange(start,min(start+rowsize,n))
        for j in range(0,n,colsize):
            cols=np.arange(j,min(j+colsize,n))
            part=blocks_param(rows,cols,chunk)
            for kind in kinds:out[kind][start:start+len(rows),j:j+len(cols)]=part[kind]
    starts=list(range(0,n,rowsize))
    if threads==1:
        for s in starts:row_block(s)
    else:
        with ThreadPoolExecutor(max_workers=threads) as pool:list(pool.map(row_block,starts))
    return out

reference=None
print('%9s %8s %8s %8s %12s'%('chunk','1 thr','4 thr','speedup','peak scratch'))
for chunk in (4096,16384,32768,65536,131072,262144):
    assemble(chunk,1);gc.collect()
    t1=time.perf_counter();assemble(chunk,1);t1=time.perf_counter()-t1
    gc.collect();tracemalloc.start()
    t4=time.perf_counter();out=assemble(chunk,4);t4=time.perf_counter()-t4
    peak=tracemalloc.get_traced_memory()[1]/1024**2;tracemalloc.stop()
    matrices=2*16*n*n/1024**2
    if reference is None:reference=out
    else:
        for kind in kinds:assert np.array_equal(out[kind],reference[kind]),'chunk %d changed results'%chunk
    print('%9d %7.3fs %7.3fs %7.2fx %9.1f MiB'%(chunk,t1,t4,t1/t4,peak-matrices))
print('results identical across every chunk size: yes')
