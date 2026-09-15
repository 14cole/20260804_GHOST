"""Rewrite must be bit-identical and must actually scale."""
import sys,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np,tracemalloc
from ghost_backend.execution.options import execution_scope
from ghost_backend.twod.pulse import coefficients as C
from ghost_backend.twod.pulse.kernel import PulseKernel,PulseSystem
from ghost_backend.twod.pulse.runtime import PulseOracle,dense_matrix,_row_block_rows
from test_fmm_efficiency import mesh_for

print('chunk_pairs: order 6 -> %d, order 8 -> %d, order 32 -> %d'%(
    C.chunk_pairs(6),C.chunk_pairs(8),C.chunk_pairs(32)))

mesh,k=mesh_for('reentrant',2048,3.)
def oracle_for(kinds):
    f=PulseKernel(mesh,k);a=PulseSystem(f.n)
    for kind in kinds:a.add(f,kind)
    a.pulse_weights={kind:np.ones(f.n) for kind in kinds}
    return PulseOracle(a),f

for kinds in (('S','KP'),('S','KP','K')):
    oracle,f=oracle_for(kinds)
    n=oracle.n
    # reference: the old fixed 32x512 tiling with a 4096-pair chunk
    old_chunk=C.chunk_pairs
    C.chunk_pairs=lambda order,kinds=3:4096
    reference=np.empty((n,n),complex,order='F')
    for i in range(0,n,32):
        rows=np.arange(i,min(i+32,n))
        for j in range(0,n,512):
            cols=np.arange(j,min(j+512,n))
            reference[i:i+len(rows),j:j+len(cols)]=oracle.get_with_error(rows,cols)[0]
    C.chunk_pairs=old_chunk
    with execution_scope(dict(assembly_threads=4)):
        span=_row_block_rows(oracle,n,4)
        gc.collect();tracemalloc.start()
        new=dense_matrix(oracle,lambda:None)
        peak=tracemalloc.get_traced_memory()[1]/1024**2;tracemalloc.stop()
    same=np.array_equal(new,reference)
    print('kinds=%-14s span=%d rows  bit-identical to old tiling: %s  scratch %.1f MiB over 4 threads (%.1f/thread)'%(
        '+'.join(kinds),span,same,peak-16*n*n/1024**2,(peak-16*n*n/1024**2)/4))
    assert same
    f.native_plan=None

oracle,f=oracle_for(('S','KP'))
for threads in (1,2,4):
    with execution_scope(dict(assembly_threads=threads)):
        dense_matrix(oracle,lambda:None)
        t=time.perf_counter();dense_matrix(oracle,lambda:None);e=time.perf_counter()-t
    if threads==1:base=e
    print('dense_matrix threads=%d  %6.3f s  (%.2fx)'%(threads,e,base/e))
