"""Equilibrated sparse-preconditioned GMRES with explicit residual rejection."""
import numpy as np
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
        requested_recycle=option('fmm_recycle_vectors','auto')
        self.recycle_limit=(getattr(matrix,'preferred_recycle_vectors',12)
                            if requested_recycle=='auto' else requested_recycle)
        self.recycled=[]
        self.coordinates=kwargs.get('coordinates')
        self.coarse_attempted=False
        self.coarse_trial=None
        from ghost_backend.twod.fmm.memory import rhs_basis_capacity
        self.rhs_basis_capacity=rhs_basis_capacity(len(matrix))
        self.rhs_q=np.empty((len(matrix),0),complex)
        self.rhs_x=np.empty((len(matrix),0),complex)
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
            recycle_vectors_requested=requested_recycle,
            rhs_basis_capacity=self.rhs_basis_capacity,rhs_basis_columns=0,rhs_basis_reused_batches=0,
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

    @timed_stage('fmm_coarse_setup')
    def _coarse_correction(self):
        """A bounded spatial coarse solve supplements the near-field ILU.

        P = M + (Z-MAZ)(Z*AZ)^-1 Z*. No operator is approximated here;
        original-equation residual checks still decide acceptance.
        """
        self.coarse_attempted=True
        n=len(self.a)
        coordinates=np.asarray(self.coordinates)
        if coordinates.shape != (n,2) or not np.all(np.isfinite(coordinates)):
            return False
        groups=[np.arange(n)]
        while len(groups)<16:
            index=max(range(len(groups)),key=lambda j:len(groups[j]))
            ids=groups.pop(index)
            if len(ids)<4:return False
            axis=int(np.argmax(np.ptp(coordinates[ids],axis=0)))
            ids=ids[np.argsort(coordinates[ids,axis],kind='stable')]
            groups.extend((ids[:len(ids)//2],ids[len(ids)//2:]))
        z=np.zeros((n,len(groups)),complex)
        for j,ids in enumerate(groups):z[ids,j]=1/np.sqrt(len(ids))
        self.checkpoint()
        az=self.eq@z
        e=z.conj().T@az
        if not np.all(np.isfinite(e)) or np.linalg.cond(e)>1e8:
            self.details['coarse_rejection']='Ill-conditioned coarse system'
            return False
        inv=np.linalg.inv(e)
        lu=self.lu
        q=z-lu.solve(az)
        previous=(self.pre,self.preh,list(self.recycled))
        self.pre=LinearOperator((n,n),dtype=complex,
            matvec=lambda b:lu.solve(b)+q@(inv@(z.conj().T@b)))
        self.preh=LinearOperator((n,n),dtype=complex,
            matvec=lambda b:lu.solve(b,trans='H')+z@(inv.conj().T@(q.conj().T@b)))
        # Keep the useful directions, but invalidate products of the old
        # preconditioner. LGMRES recomputes entries whose product is None.
        self.recycled[:]=[(v,None) for v,_ in self.recycled]
        if self.details['iterations']:
            self.coarse_trial=(previous,self.details['iterations'][-1])
        self.details.update(preconditioner='equilibrated_ilu_with_spatial_coarse_correction',
            coarse_vectors=len(groups),coarse_storage_bytes=z.nbytes+q.nbytes+inv.nbytes,
            coarse_setup_operator_columns=len(groups))
        return True

    def _finish_coarse_trial(self,iterations,remaining,failure=None):
        if self.coarse_trial is None:return
        previous,reference=self.coarse_trial
        self.coarse_trial=None
        # Demand an observed reduction with enough remaining work to repay
        # setup. This is a work estimate, not a promise about later angles.
        if failure is None and iterations<.95*reference and (reference-iterations)*remaining>=16:
            self.details['coarse_trial_accepted']=True
            return
        self.pre,self.preh,recycled=previous
        self.recycled[:]=recycled
        self.details.update(preconditioner='equilibrated_sparse_ilu',
            coarse_trial_accepted=False,coarse_storage_bytes=0,
            coarse_rejection=failure or 'Observed iteration savings do not repay setup')

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
        recovery=None;to_solve=scaled;cached=None;basis=None
        compression=option('rhs_compression','auto')
        existing=self.rhs_q.shape[1]
        if compression!='off' and (existing or b.shape[1]>=(2 if compression=='on' else 16)) and np.any(scaled):
            from ghost_backend.linalg.sweep import _qr_basis
            threshold=min(1e-12,self.tolerance*.01)*np.linalg.norm(scaled)
            cached=self.rhs_q.conj().T@scaled
            remainder=scaled-self.rhs_q@cached
            correction=self.rhs_q.conj().T@remainder
            cached+=correction;remainder-=self.rhs_q@correction
            limit=min(self.rhs_basis_capacity-existing,int(np.ceil(.7*b.shape[1]))-1)
            basis,recovery=_qr_basis(remainder,threshold,max_rank=limit)
            if basis is None and existing:
                # The old angular span is full or no longer useful. Bound the
                # cache before constructing a replacement for this batch.
                self.rhs_q=self.rhs_q[:,:0].copy();self.rhs_x=self.rhs_x[:,:0].copy()
                existing=0;cached=np.empty((0,b.shape[1]),complex)
                basis,recovery=_qr_basis(scaled,threshold,
                    max_rank=min(self.rhs_basis_capacity,int(np.ceil(.7*b.shape[1]))-1))
            if basis is not None:to_solve=basis
        solutions=[]
        for j,column in enumerate(to_solve.T):
            remaining=to_solve.shape[1]-j-1
            try:solution=self._solve_eq(column)
            except FMMConvergenceError:
                if self.coarse_trial is None:raise
                self._finish_coarse_trial(0,remaining,failure='Trial failed the checked solve; restored ILU')
                solution=self._solve_eq(column)
            solutions.append(solution)
            self._finish_coarse_trial(self.details['iterations'][-1],remaining)
            if (getattr(self.a,'spatial_coarse_eligible',False) and
                    not self.coarse_attempted and len(self.a)>=128 and
                    remaining>=8 and self.details['iterations'][-1]>=24):
                self._coarse_correction()
        solved=np.column_stack(solutions) if solutions else np.empty((len(self.a),0),complex)
        x=(self.rhs_x@cached+solved@recovery) if recovery is not None else solved
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
        if recovery is not None:
            self.rhs_q=np.column_stack((self.rhs_q,basis))
            self.rhs_x=np.column_stack((self.rhs_x,solved))
            self.details['rhs_basis_reused_batches']+=int(existing>0)
        self.details['rhs_basis_columns']=self.rhs_q.shape[1]
        self.details['rhs_basis_storage_bytes']=self.rhs_q.nbytes+self.rhs_x.nbytes
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
