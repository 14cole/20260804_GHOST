"""How much does per-call array size drive the thread scaling?"""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
from ghost_backend.twod.pulse import coefficients as C
from test_fmm_efficiency import mesh_for

mesh,k=mesh_for('reentrant',2048,3.);g=AssemblyGeometry(mesh);k=complex(k)
n=len(g.lengths);kinds={'S','KP'};order=6

def blocks_param(rows,cols,chunk,out):
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
    return res

def assemble(tile_rows,tile_cols,chunk,threads):
    out=np.empty((n,n),complex,order='F')
    def row_block(start):
        rows=np.arange(start,min(start+tile_rows,n))
        for j in range(0,n,tile_cols):
            cols=np.arange(j,min(j+tile_cols,n))
            blocks_param(rows,cols,chunk,out)
    starts=list(range(0,n,tile_rows))
    if threads==1:
        for s in starts:row_block(s)
    else:
        with ThreadPoolExecutor(max_workers=threads) as pool:list(pool.map(row_block,starts))

print('%-28s %8s %8s   %s'%('tile rows x cols / chunk','1 thr','4 thr','speedup'))
for tr,tc,ch in ((32,512,4096),(32,512,16384),(128,2048,16384),(128,2048,262144),(256,2048,524288)):
    assemble(tr,tc,ch,1)
    t1=time.perf_counter();assemble(tr,tc,ch,1);t1=time.perf_counter()-t1
    t4=time.perf_counter();assemble(tr,tc,ch,4);t4=time.perf_counter()-t4
    print('%-28s %7.3fs %7.3fs   %.2fx'%('%dx%d / %d'%(tr,tc,ch),t1,t4,t1/t4))
