"""Keep a screened near/mid-range table; evaluate all remaining entries exactly."""
from common import *
from contextlib import contextmanager
from unittest.mock import patch
import inspect
import cpu_kernels

ORIGINAL=cpu_kernels.KernelTable
SELECTOR=cpu_kernels.select_far_kernels

class PartialTable(ORIGINAL):
    def __init__(self,k,upper):
        limited=min(upper,128/max(-complex(k).imag,1e-300))
        super().__init__(k,limited)
        self.evidence['requested_upper']=upper
        self.evidence['table_upper']=limited

def selector(mesh,k,green,hankel):
    fg,fh=SELECTOR(mesh,k,green,hankel)
    table=inspect.getclosurevars(fg).nonlocals.get('table')
    if table is None:return fg,fh
    def evaluator(channel,original):
        def evaluate(k0,real_k,dist,kr,scratch,out):
            out[:]=table.polys[channel](dist)
            bad=~np.isfinite(out)
            if bad.any():
                dd=dist[bad]
                rr=np.empty_like(dd);ww=np.empty_like(dd);oo=np.empty(dd.shape,complex)
                if real_k:ops._far_kernel_argument(k0,dd,rr)
                original(k0,real_k,dd,rr,ww,oo)
                out[bad]=oo
        return evaluate
    return evaluator(0,green),evaluator(1,hankel)

@contextmanager
def partial_table():
    with patch.object(cpu_kernels,'KernelTable',PartialTable),patch.object(cpu_kernels,'select_far_kernels',selector):yield

def probe():
    rng=np.random.RandomState(437)
    records=[]
    for frequency in (1,2,5,10):
        k=(420.85-204.63j)*frequency
        d=np.r_[rng.uniform(5e-13,.31,200000),np.geomspace(5e-13,.31,10000)]
        record=dict(frequency_ghz=frequency,k=[k.real,k.imag])
        try:
            table=ORIGINAL(k,.31);record['original_table']=table.evidence
        except cpu_kernels.Rejected as e:record['original_rejection']=str(e)
        table=PartialTable(k,.31);record['partial_table']=table.evidence
        def full():return cpu_kernels.values(k,d)
        def partial():
            out=np.stack([p(d) for p in table.polys],axis=-1)
            bad=~np.all(np.isfinite(out),axis=-1)
            out[bad]=cpu_kernels.values(k,d[bad])
            return out
        expected,record['full_timing']=timed(full,5)
        got,record['partial_timing']=timed(partial,5)
        record['relative_error']=float(np.max(abs(got-expected)/np.maximum(abs(expected),1e-280)))
        record['exact_fraction']=float(np.mean(d>table.bounds[-1]))
        records.append(record)
        print(frequency,record['full_timing']['median'],record['partial_timing']['median'],record['relative_error'],flush=True)
    write('partial-table-results.json',records)

if __name__=='__main__':probe()
