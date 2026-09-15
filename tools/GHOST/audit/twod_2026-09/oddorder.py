from common import *
import numpy as np
from ghost_backend.execution.options import execution_scope
from ghost_backend.twod.pulse.kernel import PulseKernel
from ghost_backend.twod.pulse.runtime import PulseOracle,dense_matrix
from ghost_backend.twod.pulse.kernel import PulseSystem
from test_fmm_efficiency import mesh_for

mesh,k=mesh_for('reentrant',64,3.)
for order in (6,7,8,9):
    with execution_scope(dict(fmm_quadrature_order=order)):
        f=PulseKernel(mesh,k)
        print('requested',order,'-> kernel order',f.order,'policy',f.quadrature_policy)
        # how close is the nearest quad point to a center?
        g=f.geometry
        from ghost_backend.twod.pulse.coefficients import gauss
        t,w=gauss(f.order)
        quad=(g.p0[:,None]+t[None,:,None]*g.segments[:,None]).reshape(-1,2)
        d=np.linalg.norm(quad[:,None,:]-g.centers[None,:,:],axis=-1)
        print('   min |quad - center| =',d.min())
        a=PulseSystem(f.n);a.add(f,'S');a.add(f,'KP')
        x=np.ones((f.n,1),complex)
        try:
            y=a@x
            dense=dense_matrix(PulseOracle(a),lambda:None)
            err=np.max(abs(y-dense@x))/np.max(abs(dense@x))
            print('   matvec rel err vs dense:',err, 'finite:',np.all(np.isfinite(y)))
        except Exception as e:
            print('   ERROR:',type(e).__name__,e)
        f.native_plan.close()
