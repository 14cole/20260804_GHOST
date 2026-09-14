"""Equilibrated sparse-preconditioned GMRES with explicit residual rejection."""
import numpy as np
from scipy.linalg import qr
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, spilu, gmres, lgmres, onenormest
from ghost_backend.execution.options import option
from ghost_backend.execution.metrics import timed_stage
from ghost_backend.execution.errors import BackendNumericalError


class FMMConvergenceError(BackendNumericalError):
    """A checked numerical failure eligible for an admitted automatic retry."""


class FMMFactor:
    @timed_stage('fmm_preconditioner')
    def __init__(self,matrix,diagnostics=None,label='FMM system',evidence=None,
                 checkpoint=None,**kwargs):
        self.a=matrix;self.diagnostics=diagnostics;self.label=label
        self.checkpoint=checkpoint or (lambda:None)
        self.tolerance=option('fmm_solver_tolerance',1e-9)
        self.restart=option('fmm_restart',80)
        self.maxiter=option('fmm_max_iterations',800)
        self.recycle_limit=option('fmm_recycle_vectors',12)
        self.recycled=[]
        near=matrix.sparse_near()
        self.row=1/np.maximum(abs(near).max(axis=1).toarray().ravel(),1e-300)
        scaled=diags(self.row)@near
        self.col=1/np.maximum(abs(scaled).max(axis=0).toarray().ravel(),1e-300)
        scaled=(scaled@diags(self.col)).tocsc()
        try:self.lu=spilu(scaled,drop_tol=1e-5,fill_factor=10)
        except RuntimeError as exc:
            raise FMMConvergenceError('FMM sparse preconditioner is singular.') from exc
        n=len(matrix)
        # Do not capture this factor in its own LinearOperator callbacks. That
        # cycle would retain the native/preconditioner buffers until a GC pass,
        # increasing memory across frequency sweeps.
        row,col,lu=self.row,self.col,self.lu
        def mv(x):return row*(matrix@(col*x))
        def rmv(x):return col*(matrix.H@(row*x))
        def mm(x):return row[:,None]*(matrix@(col[:,None]*x))
        def rmm(x):return col[:,None]*(matrix.H@(row[:,None]*x))
        self.eq=LinearOperator((n,n),matvec=mv,rmatvec=rmv,matmat=mm,rmatmat=rmm,dtype=complex)
        self.pre=LinearOperator((n,n),matvec=self.lu.solve,dtype=complex)
        self.preh=LinearOperator((n,n),matvec=lambda b:lu.solve(b,trans='H'),dtype=complex)
        self.details=matrix.report()
        self.details.update(solver_tolerance=self.tolerance,iterations=[],condition_iterations=0,
            iterative_method='lgmres' if self.recycle_limit else 'gmres',
            iteration_measure='operator_applications',recycle_vectors_limit=self.recycle_limit,recycle_resets=0,
            preconditioner='equilibrated_sparse_ilu',preconditioner_nnz=int(self.lu.L.nnz+self.lu.U.nnz),
            preconditioner_storage_bytes=int(20*(self.lu.L.nnz+self.lu.U.nnz)+8*(n+1)),
            input_rhs_columns=0,solved_rhs_columns=0,max_relative_residual=0.)
        self.event=dict(unknowns=n,factorizations=0,rhs_batches=0,max_rhs_columns=0,
            max_backward_error=None,max_relative_residual=0.,fmm=self.details)
        if evidence is not None:evidence.append(self.event)
        self.relative_residual=np.empty(0)
        if diagnostics is not None:
            inverse=LinearOperator((n,n),dtype=complex,
                matvec=lambda b:self._solve_eq(b,condition=True),
                rmatvec=lambda b:self._solve_eq(b,adjoint=True,condition=True))
            estimate=float(onenormest(self.eq)*onenormest(inverse))
            if not np.isfinite(estimate):raise RuntimeError('FMM condition estimate is nonfinite.')
            diagnostics.update(condition_est=estimate,condition_method='near_equilibrated_1norm_fmm_gmres_estimate')

    def _solve_eq(self,b,adjoint=False,condition=False):
        b=np.asarray(b).reshape(-1)
        if not np.any(b):return np.zeros_like(b)
        count=[0]
        last_product=[None]
        a=self.eq.H if adjoint else self.eq
        def apply(x):
            self.checkpoint()
            if count[0] >= self.maxiter:
                raise FMMConvergenceError('FMM iteration budget exhausted ({} operator applications).'.format(self.maxiter))
            count[0]+=1
            last_product[0]=a@x
            return last_product[0]
        counted=LinearOperator(a.shape,matvec=apply,dtype=complex)
        if self.recycle_limit and not condition:
            augmentation=[] if adjoint else self.recycled
            previous=[float('inf')];stalled=[0]
            bnorm=np.linalg.norm(b)
            def check_progress(x):
                # LGMRES calls back immediately after its outer residual
                # application. Reuse that product, without another FMM call.
                # Almost dependent cached vectors can break down before a new
                # Krylov direction is inserted. Discard stagnant augmentation
                # while preserving the iterate, tolerance and total work cap.
                residual=np.linalg.norm(last_product[0]-b)/bnorm
                stalled[0]=stalled[0]+1 if residual >= .999*previous[0] else 0
                previous[0]=residual
                if residual > self.tolerance*.2 and stalled[0]>=2 and augmentation:
                    augmentation.clear();stalled[0]=0
                    self.details['recycle_resets']+=1
            x,info=lgmres(counted,b,M=self.preh if adjoint else self.pre,
                rtol=self.tolerance*.2,atol=0.,maxiter=self.maxiter,
                inner_m=min(self.restart,len(b)),outer_k=min(self.recycle_limit,len(b)),
                outer_v=augmentation,store_outer_Av=True,prepend_outer_v=True,callback=check_progress)
        else:
            x,info=gmres(counted,b,M=self.preh if adjoint else self.pre,rtol=self.tolerance*.2,
                atol=0.,restart=min(self.restart,len(b)),maxiter=self.maxiter,
                callback=lambda _: self.checkpoint(),callback_type='legacy')
        residual=np.linalg.norm(a@x-b)/np.linalg.norm(b)
        if info or not np.isfinite(residual) or residual>self.tolerance:
            raise FMMConvergenceError('FMM GMRES failed: info={}, operator applications={}, residual={:.3g}; limit={:.3g}.'.format(info,count[0],residual,self.tolerance))
        if condition:self.details['condition_iterations']+=count[0]
        else:self.details['iterations'].append(count[0])
        return x

    @timed_stage('linear_solve')
    def solve(self,rhs):
        import ghost_backend.twod.solver as rcs
        b=np.asarray(rhs,complex);vector=b.ndim==1
        if vector:b=b[:,None]
        if b.shape[0]!=len(self.a) or not np.all(np.isfinite(b)):raise ValueError('Invalid FMM right hand side.')
        scaled=self.row[:,None]*b
        # Compress illuminations in the equation scaling used by GMRES. Every
        # recovered column is checked in the original, unscaled equation below.
        recovery=None;to_solve=scaled
        compression=option('rhs_compression','auto')
        if compression!='off' and b.shape[1]>=(2 if compression=='on' else 16) and np.any(scaled):
            q,r,piv=qr(scaled,mode='economic',pivoting=True,check_finite=False)
            tail=np.sqrt(np.cumsum(np.sum(abs(r)**2,axis=1)[::-1])[::-1])
            rank=max(1,int(np.count_nonzero(tail>1e-12*np.linalg.norm(scaled))))
            if rank<.7*b.shape[1]:
                to_solve=q[:,:rank]
                recovery=np.empty((rank,b.shape[1]),complex);recovery[:,piv]=r[:rank]
        x=np.column_stack([self._solve_eq(column) for column in to_solve.T])
        if recovery is not None:x=x@recovery
        x=self.col[:,None]*x
        residual=self.a@x-b
        norms=np.linalg.norm(b,axis=0)
        relative=np.linalg.norm(residual,axis=0)/np.maximum(norms,1e-300)
        for j in np.flatnonzero(relative>self.tolerance):
            correction=self._solve_eq(self.row*(-residual[:,j]))
            x[:,j]+=self.col*correction
        if np.any(relative>self.tolerance):
            residual=self.a@x-b
            relative=np.linalg.norm(residual,axis=0)/np.maximum(norms,1e-300)
        if not np.all(np.isfinite(relative)) or np.max(relative)>self.tolerance:
            raise FMMConvergenceError('FMM solution failed the unscaled original-RHS residual check.')
        self.relative_residual=relative
        self.details['input_rhs_columns']+=b.shape[1]
        self.details['solved_rhs_columns']+=to_solve.shape[1]
        self.details['max_relative_residual']=max(self.details['max_relative_residual'],float(relative.max()))
        self.details['recycle_storage_bytes']=sum(v.nbytes for pair in self.recycled for v in pair if v is not None)
        self.details['native_plan_builds']=sum(k.native_plan.builds for k in self.a.kernels)
        self.details['native_plan_storage_bytes']=sum(k.native_plan.bytes.value for k in self.a.kernels)
        self.details['native_plan_peak_storage_bytes']=sum(k.native_plan.peak_bytes for k in self.a.kernels)
        self.event['rhs_batches']+=1
        self.event['max_rhs_columns']=max(self.event['max_rhs_columns'],b.shape[1])
        self.event['max_relative_residual']=self.details['max_relative_residual']
        if self.diagnostics is not None:
            self.diagnostics.update(fmm_relative_residual=self.details['max_relative_residual'],
                                    fmm_relative_residual_limit=self.tolerance)
        rcs._record_dense_backend_event(requested='cpu',used='cpu_fmm',n=len(self.a),
            label=self.label,factorizations=0,rhs_columns=b.shape[1],fmm=self.details,
            condition_method=(self.diagnostics or {}).get('condition_method'))
        return x[:,0] if vector else x
