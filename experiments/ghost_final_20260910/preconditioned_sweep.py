"""Accurate stored operator + coarse inverse-only factor; checked physical RHS.

The preconditioner never supplies acceptance evidence. Final bounds include
the accurate operator's tile/pruning error, for every original illumination.
"""
from streamed_operator import *
from checked_compressed_system import CompressedSystem
from sweep_compression import _qr_basis,_frobenius_norm
from scipy.sparse.linalg import LinearOperator,gmres


class PreconditionedSweep:
    def __init__(self,operator,coordinates,tolerance=1e-6,budget=512*1024**2,checkpoint=None):
        if not np.isfinite(budget) or budget<=0:raise ValueError('Invalid storage budget.')
        if budget<=operator.bytes:raise MemoryError('No storage remains for a preconditioner.')
        self.operator=operator;self.checkpoint=checkpoint or (lambda:None)
        self.factor=CompressedSystem(operator,coordinates,tolerance=tolerance,
            budget=budget-operator.bytes,checkpoint=self.checkpoint,inverse_only=True)
        self.norm=float(np.max(operator.row_norm))
        self.lower_norm=max(float(np.max(operator.row_norm-operator.row_error)),0.)
        self.error=float(np.max(operator.row_error))
        self.evidence=dict(preconditioner=self.factor.evidence,
            combined_bytes=operator.bytes+self.factor.bytes,refinement_steps=[],gmres_columns=0,
            physical_repairs=0,operator_products=0)

    def product(self,x):
        self.evidence['operator_products']+=1
        return self.operator.matmul(x)

    def columns(self,b):
        x=self.factor.apply(b,solve=True)
        previous=np.inf
        for step in range(12):
            self.checkpoint()
            residual=b-self.product(x)
            den=np.maximum(self.norm*np.max(abs(x),axis=0)+np.max(abs(b),axis=0),1e-300)
            errors=np.max(abs(residual),axis=0)/den
            bad=~np.isfinite(errors)|(errors>2e-14)
            if not np.any(bad):
                self.evidence['refinement_steps'].append(step)
                return x
            worst=float(np.max(errors))
            if not np.isfinite(worst) or (step>1 and worst>previous*1.2):break
            previous=worst
            x[:,bad]+=self.factor.apply(residual[:,bad],solve=True)
        self.evidence['gmres_columns']+=int(np.sum(bad))
        action=LinearOperator((self.operator.n,self.operator.n),matvec=self.product,dtype=complex)
        inverse=LinearOperator(action.shape,matvec=lambda v:self.factor.apply(v,solve=True),dtype=complex)
        for j in np.flatnonzero(bad):
            self.checkpoint()
            x[:,j],info=gmres(action,b[:,j],x0=x[:,j],M=inverse,rtol=1e-12,atol=0,
                restart=30,maxiter=4,callback=lambda _:self.checkpoint(),callback_type='pr_norm')
            if info:raise RuntimeError('Bounded preconditioned GMRES did not converge.')
        return x

    def solve(self,rhs):
        b=np.asarray(rhs,complex);vector=b.ndim==1
        if vector:b=b[:,None]
        if b.ndim!=2 or b.shape[0]!=self.operator.n or not 0<b.shape[1]<=512 or not np.all(np.isfinite(b)):
            raise ValueError('Expected a finite RHS batch with 1..512 columns.')
        scale=np.max(abs(b),axis=1);scale[scale==0]=1
        scaled=b/scale[:,None]
        q,recovery=_qr_basis(scaled,2e-15*max(_frobenius_norm(scaled),1e-300))
        self.evidence['basis_columns']=q.shape[1]
        if q.shape[1]<.7*b.shape[1]:
            x=self.columns(q*scale[:,None])@recovery if q.shape[1] else np.zeros_like(b)
        else:x=self.columns(b)
        q=recovery=scaled=None
        x[:,np.all(b==0,axis=0)]=0
        for attempt in range(2):
            residual=b-self.product(x)
            den=np.maximum(self.lower_norm*np.max(abs(x),axis=0)+np.max(abs(b),axis=0),1e-300)
            bounds=(np.max(abs(residual),axis=0)+self.error*np.max(abs(x),axis=0))/den
            bad=~np.isfinite(bounds)|(bounds>1e-12)
            if not np.any(bad):
                self.evidence['original_residual_bound']=float(np.max(bounds))
                return x[:,0] if vector else x
            if attempt==0:
                self.evidence['physical_repairs']+=int(np.sum(bad))
                x[:,bad]=self.columns(b[:,bad])
        raise RuntimeError('Accurate operator plus coefficient error failed physical-RHS acceptance.')
