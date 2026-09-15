import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np
from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
from ghost_backend.twod.pulse import coefficients as C
from ghost_backend.twod.fmm.galerkin import near_pairs as enumerate_near
from test_fmm_efficiency import mesh_for

mesh,k=mesh_for('reentrant',2048,3.);g=AssemblyGeometry(mesh);k=float(np.real(k))
pairs=enumerate_near(g,2*1024**3)
rows=[];cols=[]
for i,j in pairs:
    rows.append(i);cols.append(j)
    if i!=j:rows.append(j);cols.append(i)
rows=np.asarray(rows);cols=np.asarray(cols)
print('%d panels -> %d directed near pairs (%.1f per panel)'%(len(g.lengths),len(rows),len(rows)/len(g.lengths)))

def one_batch():
    return C.accurate_pairs(g,k,rows,cols,{'S','KP'})
def in_tiles(size):
    def run():
        for s in range(0,len(rows),size):
            C.accurate_pairs(g,k,rows[s:s+size],cols[s:s+size],{'S','KP'})
    return run
def in_chunks_1024():
    for s in range(0,len(rows),1024):
        C.accurate_pairs(g,k,rows[s:s+1024],cols[s:s+1024],{'S','KP'})

for name,fn in (('one call, all pairs',one_batch),
                ('batches of 1024',in_chunks_1024),
                ('~tile sized (128)',in_tiles(128)),
                ('~tile sized (216)',in_tiles(216))):
    fn();t=time.perf_counter();fn();print('  %-22s %6.3f s'%(name,time.perf_counter()-t))
