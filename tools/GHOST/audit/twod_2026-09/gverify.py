"""Galerkin near-path vectorization: identical matrices, and does it scale?"""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np
import ghost_backend.twod.operators as ops
from ghost_backend.twod import solver as s
from test_fmm_efficiency import mesh_for

def legacy_positions(elements,obs_idx,src_idx,obs_order,src_order):
    out={}
    for pos in range(obs_idx.size):
        o=elements[int(obs_idx[pos])];c=elements[int(src_idx[pos])]
        if o.panel_index==c.panel_index:continue
        if ops._linear_shared_interval_endpoint_info(o,(0.,1.),c,(0.,1.),tol=1e-9) is not None:continue
        if ops.requires_adaptive(o,c):continue
        d=float(np.linalg.norm(o.center-c.center));sc=max(o.length,c.length,ops.EPS)
        a,_=ops._near_singular_scheme(d,sc)
        q=max(int(max(obs_order,src_order)),min(16,int(max(5,a))))
        out.setdefault(q,[]).append(pos)
    return {q:np.asarray(v,dtype=np.int64) for q,v in out.items()}

for name,panels in (('reentrant',512),('reentrant',2048)):
    mesh,k=mesh_for(name,panels,3.)
    elements=list(mesh.elements)
    from ghost_backend.twod.fmm.galerkin import near_pairs as enumerate_near
    from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
    g=AssemblyGeometry(mesh)
    pairs=enumerate_near(g,2*1024**3)
    obs=np.asarray([i for i,j in pairs]+[j for i,j in pairs if i!=j],dtype=np.int64)
    src=np.asarray([j for i,j in pairs]+[i for i,j in pairs if i!=j],dtype=np.int64)
    order=np.argsort(obs*len(elements)+src,kind='mergesort');obs,src=obs[order],src[order]
    want=legacy_positions(elements,obs,src,8,8)
    got=ops._near_fixed_order_positions(
        np.asarray([e.panel_index for e in elements],dtype=np.int64),obs,src,
        g.p0,g.p0+g.segments,g.centers,g.lengths,8,8)
    assert set(want)==set(got),(sorted(want),sorted(got))
    for q in want:np.testing.assert_array_equal(got[q],want[q])
    print('%s %5d panels: %d near pairs, buckets %s -- classification identical'%(
        name,panels,obs.size,{q:len(v) for q,v in sorted(want.items())}))
