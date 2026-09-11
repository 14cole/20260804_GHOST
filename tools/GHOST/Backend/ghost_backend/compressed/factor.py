"""Checked inverse of a compressed operator, including adjoint evidence."""
import inspect
import numpy as np
from scipy.sparse.linalg import LinearOperator,gmres,onenormest
from ghost_backend.compressed.inverse import CompressedSystem
from ghost_backend.linalg.hierarchical import HierarchicalRejected
from ghost_backend.execution.metrics import timed_stage


class CompressedFactor:
    def __init__(self,operator,diagnostics=None,label='compressed system',evidence=None,checkpoint=None,**kwargs):
        from ghost_backend.compressed.runtime import storage_budget
        from ghost_backend.linalg.refined_lu import requested_precision
        if requested_precision()!='double':raise ValueError('Compressed factorization requires double precision.')
        self.a=operator;self.diagnostics=diagnostics;self.label=label
        self.checkpoint=checkpoint or operator.checkpoint
        self.matrix_inf=max(float(np.max(operator.row_norm-operator.row_error)),0.)
        self.relative_residual=np.empty(0)
        self.factor=None;self.reported=0
        self.budget=storage_budget()-operator.bytes-getattr(operator,'reserved_partner_bytes',0)
        if self.budget<=0:raise MemoryError('No compressed inverse storage remains.')
        self.event=dict(unknowns=operator.n,factorizations=0,rhs_batches=0,max_rhs_columns=0,
            max_backward_error=0.,max_relative_residual=0.,compressed=operator.evidence,
            preconditioners=[],gmres_columns=0,max_refinements=0)
        if evidence is not None:evidence.append(self.event)
        rejected=False
        try:self._build(1e-6)
        except (HierarchicalRejected,np.linalg.LinAlgError,RuntimeWarning) as exc:
            self.event['coarse_rejection']=str(exc);rejected=True
        if rejected:self._build(2e-10)
        if diagnostics is not None:self._condition()

    @timed_stage('factorization')
    def _build(self,tolerance):
        self.factor=None
        self.checkpoint()
        self.factor=CompressedSystem(self.a,self.a.coordinates,tolerance=tolerance,
            budget=self.budget,checkpoint=self.checkpoint,inverse_only=True)
        self.tolerance=tolerance;self.event['factorizations']+=1
        self.event['preconditioners'].append(self.factor.evidence)

    def physical_errors(self,x,b,residual=None,trans=0):
        if residual is None:residual=self.a.matmul(x,trans)-b
        rows=self.a.row_error if trans==0 else self.a.column_error
        norm=self.a.row_norm if trans==0 else self.a.column_norm
        lower=max(float(np.max(norm-rows)),0.)
        magnitude=np.max(abs(x),axis=0)
        numerator=np.max(abs(residual)+rows[:,None]*magnitude[None,:],axis=0)
        den=np.maximum(lower*magnitude+np.max(abs(b),axis=0),1e-300)
        return numerator/den

    def relative_errors(self,x,b,residual):
        error=abs(residual)+self.a.row_error[:,None]*np.max(abs(x),axis=0)[None,:]
        norm=np.linalg.norm(b,axis=0)
        from ghost_backend.twod.constants import EPS
        return np.linalg.norm(error,axis=0)/np.where(norm<=EPS,1.,norm)

    def _refine(self,b,trans):
        x=self.factor.apply(b,solve=True,trans=trans)
        norm=float(np.max(self.a.row_norm if trans==0 else self.a.column_norm))
        previous=np.inf
        for step in range(10):
            self.checkpoint();residual=b-self.a.matmul(x,trans)
            denominator=np.maximum(norm*np.max(abs(x),axis=0)+np.max(abs(b),axis=0),1e-300)
            errors=np.max(abs(residual),axis=0)/denominator
            bad=~np.isfinite(errors)|(errors>3e-15)
            if not np.any(bad):
                self.event['max_refinements']=max(self.event['max_refinements'],step)
                return x,-residual
            worst=float(np.max(errors))
            if step==9 or not np.isfinite(worst) or step>1 and worst>previous*1.2:break
            previous=worst;x[:,bad]+=self.factor.apply(residual[:,bad],solve=True,trans=trans)
        self.event['gmres_columns']+=int(np.sum(bad))
        action=LinearOperator(self.a.shape,matvec=lambda z:self.a.matmul(z,trans),dtype=complex)
        inverse=LinearOperator(self.a.shape,matvec=lambda z:self.factor.apply(z,solve=True,trans=trans),dtype=complex)
        parameters=inspect.signature(gmres).parameters
        modern='rtol' in parameters
        for j in np.flatnonzero(bad):
            iterations=[0]
            def callback(value):
                self.checkpoint();iterations[0]+=1
                if iterations[0]>120:raise HierarchicalRejected('Compressed GMRES iteration cap exceeded.')
            params=dict(x0=x[:,j],M=inverse,restart=30,maxiter=4 if modern else 120,callback=callback)
            params.update(dict(rtol=1e-13,atol=0.,callback_type='pr_norm') if modern else dict(tol=1e-13))
            if 'atol' in parameters:params['atol']=0.
            if 'callback_type' in parameters:params.update(callback_type='pr_norm',maxiter=4)
            x[:,j],info=gmres(action,b[:,j],**params)
            if info:raise HierarchicalRejected('Compressed GMRES did not converge within its cap.')
        return x,self.a.matmul(x,trans)-b

    def inverse(self,rhs,trans=0,return_residual=False):
        if trans not in (0,1,2):raise ValueError('Invalid transpose mode.')
        b=np.asarray(rhs,complex);vector=b.ndim==1
        if vector:b=b[:,None]
        if b.ndim!=2 or b.shape[0]!=len(self.a) or not b.shape[1] or not np.all(np.isfinite(b)):
            raise ValueError('Invalid compressed RHS.')
        failed=False
        try:x,residual=self._refine(b,trans)
        except (HierarchicalRejected,np.linalg.LinAlgError,RuntimeWarning) as exc:
            if self.tolerance<=2e-10:raise
            self.event['coarse_rejection']=str(exc);failed=True
        if failed:
            self._build(2e-10);x,residual=self._refine(b,trans)
        errors=self.physical_errors(x,b,residual,trans)
        if not np.all(np.isfinite(errors)) or np.max(errors)>1e-12:
            raise HierarchicalRejected('Compressed inverse failed the original-coefficient error bound.')
        self.event['max_backward_error']=max(self.event['max_backward_error'],float(np.max(errors)))
        solution=x[:,0] if vector else x
        if return_residual:return solution,residual[:,0] if vector else residual
        return solution

    def _condition(self):
        rows,columns,norm=self.a.equilibrate()
        inverse=LinearOperator(self.a.shape,
            matvec=lambda z:columns*self.inverse(rows*np.asarray(z).reshape(-1)),
            rmatvec=lambda z:rows*self.inverse(columns*np.asarray(z).reshape(-1),trans=2),dtype=complex)
        estimate=norm*float(onenormest(inverse))
        if not np.isfinite(estimate):raise HierarchicalRejected('Nonfinite compressed condition estimate.')
        self.diagnostics.update(condition_est=estimate,condition_method='equilibrated_1norm_compressed_refined_inverse',condition_label=self.label)

    @timed_stage('linear_solve')
    def solve(self,rhs):
        import ghost_backend.twod.solver as rcs
        b=np.asarray(rhs,complex);vector=b.ndim==1
        if vector:b=b[:,None]
        x,residual=self.inverse(b,return_residual=True)
        self.relative_residual=self.relative_errors(x,b,residual)
        self.event['rhs_batches']+=1
        self.event['max_rhs_columns']=max(self.event['max_rhs_columns'],b.shape[1])
        self.event['max_relative_residual']=max(self.event['max_relative_residual'],float(np.max(self.relative_residual)))
        if self.diagnostics is not None:
            self.diagnostics.update(linear_backward_error=self.event['max_backward_error'],linear_backward_error_limit=1e-12)
        rcs._record_dense_backend_event(requested='cpu',used='cpu_compressed',n=len(self.a),label=self.label,
            factorizations=self.event['factorizations']-self.reported,rhs_columns=b.shape[1],compressed=self.event,
            sweep_compression=self.event.get('sweep_compression'),condition_method=(self.diagnostics or {}).get('condition_method'))
        self.reported=self.event['factorizations']
        return x[:,0] if vector else x
