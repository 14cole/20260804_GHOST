"""Recheck real-coefficient complex accumulation on this installed BLAS."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='2'
import time,json
from pathlib import Path
import numpy as np
from scipy.linalg.blas import daxpy,zaxpy
records=[]
for n in (4096,16384,65536,262144,1048576):
    x=np.full(n,1.3+.2j);y=np.zeros(n,complex);tmp=np.empty_like(x)
    def normal():np.multiply(x,.17,out=tmp);np.add(y,tmp,out=y)
    def real():daxpy(x.view(float),y.view(float),a=.17)
    def complex_():zaxpy(x,y,a=.17)
    record=dict(entries=n)
    for name,fn in (('numpy',normal),('daxpy',real),('zaxpy',complex_)):
        fn();runs=[]
        for _ in range(5):
            y.fill(0);start=time.perf_counter()
            for i in range(40):fn()
            runs.append((time.perf_counter()-start)/40)
        np.testing.assert_allclose(y,40*.17*x,rtol=1e-14,atol=1e-14)
        record[name]=float(np.median(runs))
    records.append(record)
Path(__file__).with_suffix('.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
