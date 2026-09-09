"""One FP64 LU, bounded RHS blocks, original condition/backward-error gates."""
import rcs_solver as rcs
from cpu_execution import current_state
import time
import numpy as np
import scipy.linalg as la
from solver_metrics import timed_stage


class CheckedLU:
    def __init__(self,a,diagnostics,label,evidence):
        current_state().checkpoint()
        self.a=np.asarray(a,dtype=np.complex128);self.diagnostics=diagnostics;self.label=label
        rcs._ensure_finite_linear_system(self.a,label=label)
        self.lu,self.piv=timed_stage("factorization")(la.lu_factor)(self.a,check_finite=False)
        if diagnostics is not None:
            diagnostics.update(condition_est=rcs._equilibrated_condition_from_lu(self.a,self.lu,self.piv),
                condition_method="equilibrated_1norm_lu_onenormest",condition_label=label,
                linear_backward_error=0.,linear_backward_error_limit=rcs.DENSE_LINEAR_BACKWARD_ERROR_MAX,linear_refinement_steps=0)
        # Same row sums as the original infinity norm, bounded real workspace.
        self.matrix_inf=max(float(np.max(np.sum(np.abs(self.a[i:i+64]),axis=1))) for i in range(0,len(self.a),64))
        self.relative_residual=np.empty(0);self.event=dict(unknowns=len(a),factorizations=1,rhs_batches=0,max_rhs_columns=0,
            max_backward_error=0.,max_relative_residual=0.)
        evidence.append(self.event)

    @timed_stage("linear_solve")
    def solve(self,b):
        current_state().checkpoint()
        b=np.asarray(b,dtype=np.complex128)
        if not np.all(np.isfinite(b)):raise ValueError("Nonfinite RHS")
        x=timed_stage("rhs_solve")(la.lu_solve)((self.lu,self.piv),b,check_finite=False)
        def metrics(value):
            residual=self.a@value-b
            ri=np.max(np.abs(residual),axis=0)
            den=self.matrix_inf*np.max(np.abs(value),axis=0)+np.max(np.abs(b),axis=0)
            errors=np.divide(ri,den,out=np.zeros_like(ri,dtype=float),where=den>0)
            errors[(den<=0)&(ri>0)]=np.inf
            return residual,float(np.max(errors))
        residual,error=metrics(x);steps=0;limit=rcs.DENSE_LINEAR_BACKWARD_ERROR_MAX
        for attempt in range(2):
            if error<=limit:break
            candidate=x+la.lu_solve((self.lu,self.piv),-residual,check_finite=False)
            r2,e2=metrics(candidate)
            if e2>=error:break
            x,residual,error=candidate,r2,e2;steps+=1
        if not np.isfinite(error) or error>limit:raise RuntimeError(f"{self.label} backward error {error:g} exceeds original limit {limit:g}")
        norm=np.linalg.norm(b,axis=0);norm=np.where(norm<=rcs.EPS,1.,norm)
        self.relative_residual=np.linalg.norm(residual,axis=0)/norm
        self.event["rhs_batches"]+=1;self.event["max_rhs_columns"]=max(self.event["max_rhs_columns"],b.shape[1])
        self.event["max_backward_error"]=max(self.event["max_backward_error"],error)
        self.event["max_relative_residual"]=max(self.event["max_relative_residual"],float(np.max(self.relative_residual)))
        if self.diagnostics is not None:
            self.diagnostics["linear_backward_error"]=max(self.diagnostics["linear_backward_error"],error)
            self.diagnostics["linear_refinement_steps"]=max(self.diagnostics["linear_refinement_steps"],steps)
        rcs._record_dense_backend_event(requested="cpu",used="cpu",n=len(self.a),label=self.label,
            fallback_reason="",gpu_device="",refinement_steps=steps)
        return x
