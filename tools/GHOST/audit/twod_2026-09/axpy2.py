import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT)]
import numpy as np, inspect
from scipy.linalg.blas import zaxpy, daxpy
from ghost_backend.twod.operators import _axpy_into
print('zaxpy accepts overwrite_y:', 'overwrite_y' in str(inspect.signature(zaxpy)) if hasattr(zaxpy,'__doc__') else '?')
print('zaxpy doc sig:', zaxpy.__doc__.splitlines()[0].strip())
print('daxpy doc sig:', daxpy.__doc__.splitlines()[0].strip())

tile=284; reps=400; c=0.37
rng=np.random.default_rng(0)
src=rng.normal(size=(tile,tile))+1j*rng.normal(size=(tile,tile))
acc=np.zeros((tile,tile),complex); scratch=np.empty_like(acc)

def two_ufunc():
    for _ in range(reps): _axpy_into(acc,src,c,scratch)
def blas_writeback():
    fa=acc.reshape(-1); fs=src.reshape(-1)
    for _ in range(reps): fa[:]=zaxpy(fs,fa,a=c)
def real_view():
    # complex acc with a REAL coefficient is two real axpys over one buffer
    fa=acc.view(float).reshape(-1); fs=src.view(float).reshape(-1)
    for _ in range(reps): fa[:]=daxpy(fs,fa,a=c)

for name,fn in (('two ufuncs (current)',two_ufunc),('zaxpy + write-back',blas_writeback),
                ('daxpy on real view + write-back',real_view)):
    acc[:]=0; fn(); acc[:]=0
    t=time.perf_counter(); fn(); e=time.perf_counter()-t
    print('  %-32s %6.3f ms per call'%(name,1e3*e/reps))
a1=np.zeros((tile,tile),complex);_axpy_into(a1,src,c,np.empty_like(a1))
a2=np.zeros((tile,tile),complex);f=a2.reshape(-1);f[:]=zaxpy(src.reshape(-1),f,a=c)
print('  zaxpy path identical:',np.array_equal(a1,a2))
