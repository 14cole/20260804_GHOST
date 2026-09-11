"""Bounded GMRES counterfactual using dense A as an exact matvec oracle.

This is an iteration-count/preconditioning test, not an FMM memory benchmark.
"""
from common import *
import warnings
from unittest.mock import patch
import scipy.linalg as la
from scipy.sparse.linalg import gmres,LinearOperator
import hierarchical_factor as hf
from algebra import project

def run(folder,pol):
    prefix=HERE/'systems'/folder/pol
    a=np.load(str(prefix)+'-a.npy');xy=np.load(str(prefix)+'-xy.npy')
    b=np.load(str(prefix)+'-b.npy')[:,[0,43,90,180]]
    reference_full=np.load(str(prefix)+'-x.npy')
    reference=reference_full[:,[0,43,90,180]]
    reference_field=project(prefix,reference_full)
    scale=np.maximum(np.max(abs(a),axis=1),1e-300);norm=hf.matrix_inf_norm(a)
    operator=LinearOperator(a.shape,matvec=lambda x:(a@x)/scale,dtype=complex)
    def ordering(ids):
        if len(ids)<=128:return [ids]
        axis=int(np.argmax(np.ptp(xy[ids],axis=0)));ids=ids[np.argsort(xy[ids,axis],kind='mergesort')]
        mid=len(ids)//2;return ordering(ids[:mid])+ordering(ids[mid:])
    records=[]
    for name in ('none','local_blocks','coarse_hodlr'):
        record=dict(name=name);start=time.perf_counter()
        try:
            preconditioner=None;payload=0
            if name=='local_blocks':
                blocks=[]
                with warnings.catch_warnings():
                    warnings.simplefilter('error')
                    for ids in ordering(np.arange(len(a))):
                        block=a[np.ix_(ids,ids)]/scale[ids,None]
                        blocks.append((ids,la.lu_factor(block,check_finite=False)))
                payload=sum(lu[0].nbytes+lu[1].nbytes for ids,lu in blocks)
                def apply(v):
                    out=np.empty_like(v)
                    for ids,lu in blocks:out[ids]=la.lu_solve(lu,v[ids],check_finite=False)
                    return out
                preconditioner=LinearOperator(a.shape,matvec=apply,dtype=complex)
            if name=='coarse_hodlr':
                original=hf.compress
                def compress(block,*args,**kw):return original(block,tolerance=1e-3)
                with patch.object(hf,'compress',compress):inverse=hf.HierarchicalFactor(a,xy)
                payload=inverse.bytes
                preconditioner=LinearOperator(a.shape,matvec=lambda v:inverse._apply((v*scale)[:,None],0)[:,0],dtype=complex)
            record.update(setup_seconds=time.perf_counter()-start,preconditioner_bytes=payload,solves=[])
            solutions=[]
            for col in range(b.shape[1]):
                history=[];start=time.perf_counter()
                x,info=gmres(operator,b[:,col]/scale,M=preconditioner,restart=40,maxiter=3,
                    rtol=1e-11,atol=0,callback=history.append,callback_type='pr_norm')
                elapsed=time.perf_counter()-start
                den=norm*float(abs(x).max())+float(abs(b[:,col]).max())
                record['solves'].append(dict(info=int(info),iterations=len(history),seconds=elapsed,
                    original_backward=float(abs(a@x-b[:,col]).max()/max(den,1e-300)),
                    density_error=difference(x,reference[:,col])))
                solutions.append(x)
            changed=reference_full.copy();changed[:,[0,43,90,180]]=np.stack(solutions,axis=1)
            record['field_peak_error']=difference(project(prefix,changed),reference_field)
        except Exception as e:record['rejected']=str(e)
        records.append(record)
        print(folder,pol,name,record,flush=True)
    write('iterative-'+folder+'-'+pol+'.json',records)

if __name__=='__main__':
    for folder in ('pec-n384-f0.6','airfoil-n256-f1.0'):
        for pol in ('TE','TM'):run(folder,pol)
