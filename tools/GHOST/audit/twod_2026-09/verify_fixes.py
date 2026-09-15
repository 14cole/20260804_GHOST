import sys,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np
from scipy.integrate import quad_vec
warnings.simplefilter('error')  # surface any IntegrationWarning

# --- quad_vec replacement accuracy ---
from ghost_backend.twod.pulse.coefficients import _self_integral,green
worst=0.
for k in (0.3, 20.94, 3.0-0.4j, 12.-2j, 1e-3, 200.):
    for length in (1e-4, 1e-2, .05, .2, 1.0, 3.0):
        ref,_=quad_vec(lambda t:green(k,np.array([length*t]),False)[0][0],
                       0.,.5,epsabs=1e-15,epsrel=1e-13)
        ref=2*length*ref
        got=_self_integral(complex(k),float(length))
        rel=abs(got-ref)/max(abs(ref),1e-300)
        worst=max(worst,rel)
print('self-integral: worst relative difference vs quad_vec = %.3e'%worst)

# --- _query_balls fallback equals the vectorized path ---
from scipy.spatial import cKDTree
from ghost_backend.twod.fmm import galerkin as G
rng=np.random.default_rng(7)
pts=rng.normal(size=(400,2)); radii=rng.uniform(.05,.6,size=120)
tree=cKDTree(pts); targets=pts[:120]
G._VECTOR_RADIUS[0]=True
fast=[sorted(v) for v in G._query_balls(tree,targets,radii)]
G._VECTOR_RADIUS[0]=False
slow=[sorted(v) for v in G._query_balls(tree,targets,radii)]
G._VECTOR_RADIUS[0]=True
print('query_balls fallback identical:',fast==slow)

# --- near_pairs identical with the fallback forced ---
from test_fmm_efficiency import mesh_for
from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
for name in ('reentrant','circle'):
    try: mesh,k=mesh_for(name,96,3.)
    except Exception: continue
    g=AssemblyGeometry(mesh)
    G._VECTOR_RADIUS[0]=True;  a=G.near_pairs(g,float('inf'))
    G._VECTOR_RADIUS[0]=False; b=G.near_pairs(g,float('inf'))
    G._VECTOR_RADIUS[0]=True
    print('near_pairs(%s): %d pairs, fallback identical: %s'%(name,len(a),a==b))
