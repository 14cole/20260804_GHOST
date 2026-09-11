"""Reuse an already computed exact-A residual, without an algebraic shortcut."""
from support import *
from owned_assembly import clone
from contextlib import contextmanager
from unittest.mock import patch
import hierarchical_factor as hf
import refined_lu as mixed
import dense_factor as df
from algebra import qr_basis
STATS={}

def residual(factor,b,x):
    inner=factor.hierarchical if factor.hierarchical is not None else factor.mixed
    cached=getattr(inner,'audit_last',None)
    if inner is not None:inner.audit_last=None
    if cached is not None and cached[0] is factor.a and cached[1] is b and cached[2] is x and cached[4]==0:
        # Inner refinement owns b-Ax; the public wrapper uses Ax-b.
        value=cached[3];np.negative(value,out=value)
        STATS['hits']+=1
        return value
    STATS['misses']+=1
    return factor.a@x-b

@contextmanager
def reuse():
    STATS.clear();STATS.update(hits=0,misses=0)
    hierarchy=clone(hf.HierarchicalFactor.solve,[
        ('return x[:, 0] if vector else x',
         'self.audit_last=(self.a,b,x,residual,trans)\n                return x[:, 0] if vector else x')])
    refined=clone(mixed.RefinedLU.solve,[
        ('                return x','                self.audit_last=(self.matrix,b,x,residual,trans)\n                return x')])
    public=clone(df.DenseFactor.solve,[('residual = self.a @ candidate - b','residual = audit_residual(self,b,candidate)')],
        dict(audit_residual=residual))
    with patch.object(hf.HierarchicalFactor,'solve',hierarchy),patch.object(mixed.RefinedLU,'solve',refined),patch.object(df.DenseFactor,'solve',public):yield

def run(folder,pol):
    from contextlib import nullcontext
    prefix=PRIOR/'systems'/folder/pol
    a=np.load(str(prefix)+'-a.npy');b=np.load(str(prefix)+'-b.npy');xy=np.load(str(prefix)+'-xy.npy')
    basis,_,_=qr_basis(b)
    records=[]
    for mode,precision in (('hierarchical','double'),('dense','mixed')):
        with patch.dict(os.environ,{'GHOST_CPU_FACTORIZATION':mode}),mixed.linear_precision(precision):
            factor=df.DenseFactor(a,{},coordinates=xy)
        baseline=None
        for cache in (False,True):
            with reuse() if cache else nullcontext():
                x,t=timed(lambda:factor.solve(basis),4)
                stats=dict(STATS) if cache else {}
            if baseline is None:baseline=x
            den=np.maximum(factor.matrix_inf*np.max(abs(x),axis=0)+np.max(abs(basis),axis=0),1e-300)
            records.append(dict(factor=mode,precision=precision,reuse=cache,timing=t,cache_stats=stats,
                solution_difference=difference(x,baseline),backward=float(np.max(np.max(abs(a@x-basis),axis=0)/den)),
                mixed_used=factor.mixed is not None))
        factor=None
    write('residual-'+folder+'-'+pol+'.json',records)
    print(pol,records,flush=True)

if __name__=='__main__':
    for pol in ('TE','TM'):run('airfoil-n256-f2.0',pol)
