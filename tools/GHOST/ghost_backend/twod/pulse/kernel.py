"""P0 point-collocation FMM with distinct source and testing maps.

The native source-only plan contains Gauss points and zero-strength testing
centers. Adjoint products exchange these maps explicitly: collocation K and K'
are not matrix transposes. Native work stays bounded to eight densities.
"""
import numpy as np
from scipy.sparse import coo_matrix
from ghost_backend.execution.options import option,effective_assembly_threads
from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
from ghost_backend.twod.fmm.kernel import NativePlan,evaluate
from ghost_backend.twod.fmm.quadrature import quadrature_order
from ghost_backend.twod.fmm.galerkin import near_pairs as enumerate_near
from ghost_backend.twod.fmm.system import FMMSystem
from ghost_backend.twod.pulse.coefficients import gauss,blocks,point_pairs,accurate_pairs


class PulseKernel:
    def __init__(self,mesh,k,checkpoint=None,budget=2*1024**3):
        self.geometry=AssemblyGeometry(mesh)
        self.n=self.m=len(mesh.elements);self.k=complex(k)
        self.eps=option('fmm_tolerance',1e-10);self.threads=effective_assembly_threads()
        self.order,self.quadrature_policy=quadrature_order(k,self.geometry.lengths.max(),
            option('fmm_quadrature_order',0),8,self.eps)
        self.graded=bool(option('far_grading',True) and not option('fmm_quadrature_order',0) and self.eps>=1e-10)
        self.checkpoint=checkpoint or (lambda:None);self.budget=budget
        self.needed=set();self.near={};self.correction={};self.pairs=[]
        self.points=np.empty((0,2));self.native_plan=None;self.calls=0

    def coefficients(self,kinds,rows,cols):
        self.checkpoint()
        return blocks(self.geometry,self.k,rows,cols,kinds,self.order,self.graded)

    def prepare(self):
        missing=self.needed-set(self.near)
        if self.native_plan is not None and not missing:return
        g=self.geometry;t,w=gauss(self.order)
        if self.native_plan is None:
            quad=(g.p0[:,None]+t[None,:,None]*g.segments[:,None]).reshape(-1,2)
            self.points=np.vstack((quad,g.centers))
            self.normals=np.vstack((np.repeat(g.normals,self.order,axis=0),g.normals))
            self.weights=g.lengths[:,None]*w
            self.pairs=enumerate_near(g,self.budget)
            count=sum(1 if i==j else 2 for i,j in self.pairs)
            rows=np.empty(count,int);cols=np.empty(count,int);cursor=0
            for i,j in self.pairs:
                rows[cursor]=i;cols[cursor]=j;cursor+=1
                if i!=j:rows[cursor]=j;cols[cursor]=i;cursor+=1
            self.near_rows,self.near_cols=rows,cols
            self.native_plan=NativePlan(self.points,self.k,self.eps)
        rows,cols=self.near_rows,self.near_cols
        exact={kind:np.empty(len(rows),complex) for kind in missing}
        correction={kind:np.empty(len(rows),complex) for kind in missing}
        for start in range(0,len(rows),1024):
            self.checkpoint();stop=min(start+1024,len(rows));r=rows[start:stop];c=cols[start:stop]
            fine=accurate_pairs(g,self.k,r,c,missing)
            point=point_pairs(g,self.k,r,c,missing,self.order)
            for kind in missing:
                exact[kind][start:stop]=fine[kind]
                correction[kind][start:stop]=fine[kind]-point[kind]
        for kind in missing:
            self.near[kind]=coo_matrix((exact[kind],(rows,cols)),shape=(self.n,self.n)).tocsr()
            self.correction[kind]=coo_matrix((correction[kind],(rows,cols)),shape=(self.n,self.n)).tocsr()
            self.near[kind].eliminate_zeros();self.correction[kind].eliminate_zeros()

    def _point_apply(self,kind,x,adjoint=False,eta=None):
        self.prepare();self.checkpoint();self.calls+=1
        width=x.shape[1];nq=self.n*self.order
        strength=np.zeros((len(self.points),width),complex)
        if adjoint:strength[nq:]=x.conj()
        else:strength[:nq]=(self.weights[:,:,None]*x[:,None,:]).reshape(nq,width)
        dipole=(kind=='KP' if adjoint else kind=='K') and eta is None
        want_gradient=(kind=='K' if adjoint else kind=='KP') or (adjoint and eta is not None)
        charges=None if dipole else strength
        dipoles=strength if dipole else None
        if eta is not None and not adjoint:charges=eta*strength;dipoles=strength
        value,gradient=evaluate(self.points,self.k,charges,dipoles,self.normals,
            gradient=want_gradient,eps=self.eps,threads=self.threads,plan=self.native_plan)
        if adjoint:
            value=value[:nq]
            if want_gradient:
                normal=np.einsum('qcr,qc->qr',gradient[:nq],self.normals[:nq])
                value=normal if eta is None else normal+eta*value
            return np.sum(value.reshape(self.n,self.order,width)*self.weights[:,:,None],axis=1).conj()
        if want_gradient:return np.einsum('ncr,nc->nr',gradient[nq:],self.geometry.normals)
        return value[nq:]

    def apply(self,kind,x,mask=None,coefficient=None,adjoint=False):
        x=np.asarray(x,complex);vector=x.ndim==1
        if vector:x=x[:,None]
        if x.shape[1]>8:
            return np.column_stack([self.apply(kind,x[:,j:j+8],mask,coefficient,adjoint)
                                    for j in range(0,x.shape[1],8)])
        source=x
        if adjoint and coefficient is not None:source=x*np.asarray(coefficient).conj()[:,None]
        if not adjoint and mask is not None:source=x*np.asarray(mask)[:,None]
        y=self._point_apply(kind,source,adjoint)
        correction=self.correction[kind]
        y+=(correction.conj().T if adjoint else correction)@source
        if adjoint and mask is not None:y*=np.asarray(mask).conj()[:,None]
        if not adjoint and coefficient is not None:y*=np.asarray(coefficient)[:,None]
        return y[:,0] if vector else y

    def apply_combined(self,x,eta,adjoint=False):
        x=np.asarray(x,complex);vector=x.ndim==1
        if vector:x=x[:,None]
        if x.shape[1]>8:
            return np.column_stack([self.apply_combined(x[:,j:j+8],eta,adjoint)
                                    for j in range(0,x.shape[1],8)])
        y=self._point_apply('K',x,adjoint,eta)
        if adjoint:y+=self.correction['K'].conj().T@x+eta.conjugate()*(self.correction['S'].conj().T@x)
        else:y+=self.correction['K']@x+eta*(self.correction['S']@x)
        return y[:,0] if vector else y

    def sparse(self,kind,mask=None,coefficient=None):
        self.prepare();a=self.near[kind]
        if mask is not None:a=a.multiply(np.asarray(mask)[None,:])
        if coefficient is not None:a=a.multiply(np.asarray(coefficient)[:,None])
        return a.tocsr()

    @property
    def storage_bytes(self):
        return self.points.nbytes+sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes
            for a in list(self.near.values())+list(self.correction.values()))+sum(
                getattr(self,name,np.empty(0)).nbytes for name in ('weights','normals','near_rows','near_cols'))


class PulseSystem(FMMSystem):
    def add(self,kernel,kind,*args,**kwargs):
        kernel.needed.add(kind)
        return super().add(kernel,kind,*args,**kwargs)

    def _apply(self,x,adjoint=False):
        if hasattr(self,'_combined_kernel'):return super()._apply(x,adjoint)
        x=np.asarray(x);vector=x.ndim==1
        if vector:x=x[:,None]
        result=(self.jumps.conj().T if adjoint else self.jumps)@x
        for kernel,kind,R,C,mask,coefficient in self.terms:
            if adjoint:
                result+=C.conj().T@kernel.apply(kind,R.conj().T@x,mask,coefficient,True)
            else:result+=R@kernel.apply(kind,C@x,mask,coefficient)
        return result[:,0] if vector else result

    def report(self):
        value=super().report()
        value.update(method='pulse_fmm_gmres',discretization='pulse_collocation',
                     residual_operator='locally_corrected_fmm_pulse_collocation')
        return value
