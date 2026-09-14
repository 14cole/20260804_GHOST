"""P1 Galerkin operators: point FMM plus sparse accurate near corrections.

Corrections live on panel endpoint incidences, preserving discontinuous material
weights and source masks even at shared nodes. No dense global operator is built.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.special import hankel2
from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
from ghost_backend.twod.fmm.kernel import evaluate, NativePlan


def near_pairs(geometry, budget, count_only=False):
    """Enumerate geometric neighbours using length-binned spatial trees."""
    lengths, centers = geometry.lengths, geometry.centers
    bins = np.floor(np.log2(lengths / lengths.min())).astype(int)
    pairs = []
    count = 0
    for b in np.unique(bins):
        ids = np.flatnonzero(bins == b)
        tree = cKDTree(centers[ids])
        upper = lengths[ids].max()
        for start in range(0,len(lengths),256):
            stop = min(start+256,len(lengths))
            candidates = tree.query_ball_point(centers[start:stop],3*np.maximum(lengths[start:stop],upper))
            for i, js in enumerate(candidates,start):
                js = ids[np.asarray(js,int)]
                js = js[js>=i]
                keep = np.linalg.norm(centers[js]-centers[i],axis=1) <= 3*np.maximum(lengths[js],lengths[i])
                js = js[keep]
                count += len(js)
                # All three primitives, accurate blocks, COO/CSR construction.
                if count*2304 > budget:
                    raise MemoryError('FMM near interactions exceed the configured storage budget.')
                if not count_only:pairs.extend((i,int(j)) for j in js)
    return count if count_only else pairs


class GalerkinKernel:
    def __init__(self, mesh, k, order=8, eps=1e-10, threads=1,
                 budget=2*1024**3, checkpoint=None, geometry=None, pairs=None):
        self.mesh, self.k, self.eps, self.threads = mesh, complex(k), eps, threads
        self.checkpoint = checkpoint or (lambda: None)
        self.geometry = geometry or AssemblyGeometry(mesh)
        g=self.geometry
        if not len(g.lengths) or np.any(g.lengths<=0): raise ValueError('FMM needs nondegenerate panels.')
        self.n=len(mesh.nodes);self.m=len(g.lengths)
        self.order=max(8,int(order),int(np.ceil(abs(k)*g.lengths.max()/2))+4)
        if self.order>64: raise ValueError('FMM quadrature needs a finer mesh (order exceeds 64).')
        if 64*self.m*self.order>budget or pairs is not None and len(pairs)*2304>budget:
            raise MemoryError('FMM geometry and near workspace exceed the configured storage budget.')
        t,w=np.polynomial.legendre.leggauss(self.order);t=(t+1)/2;w=w/2
        self.phi=np.stack((1-t,t),axis=1)
        self.points=(g.p0[:,None]+t[None,:,None]*g.segments[:,None]).reshape(-1,2)
        self.native_plan=NativePlan(self.points,self.k,self.eps)
        self.points=self.native_plan.points
        self.weights=(g.lengths[:,None]*w).ravel()
        self.normals=np.repeat(g.normals,self.order,axis=0)
        self.ids=g.node_ids.ravel()
        self.P=coo_matrix((np.ones(2*self.m),(np.arange(2*self.m),self.ids)),shape=(2*self.m,self.n)).tocsr()
        self.pairs=near_pairs(g,budget) if pairs is None else pairs
        self.correction={};self.near={};self.calls=0
        self._build()

    def _build(self):
        from ghost_backend.twod.operators import _sk_blocks_near_linear
        rr=[];cc=[];delta={kind:[] for kind in ('S','KP','W')};exact={kind:[] for kind in delta}
        g=self.geometry;q=self.order;p=self.phi
        tangent=np.array([[1.,-1.],[-1.,1.]])
        points=self.points.reshape(self.m,q,2);weights=self.weights.reshape(self.m,q)
        def append(i,j,s,kp,bs,bkp):
            normal=np.dot(g.normals[i],g.normals[j])
            maue=lambda b: -self.k**2*normal*b+tangent*b.sum()/(g.lengths[i]*g.lengths[j])
            rr.extend([2*i,2*i,2*i+1,2*i+1]);cc.extend([2*j,2*j+1,2*j,2*j+1])
            for kind,a,b in (('S',s,bs),('KP',kp,bkp),('W',maue(s),maue(bs))):
                delta[kind].extend((a-b).ravel());exact[kind].extend(a.ravel())
        for num,(i,j) in enumerate(self.pairs):
            if num%64==0:self.checkpoint()
            difference=points[i,:,None]-points[j,None,:]
            r=np.linalg.norm(difference,axis=-1)
            diagonal=r==0
            safe=np.where(diagonal,1,r)
            green=.25j*hankel2(0,self.k*safe);green[diagonal]=0
            h=-.25j*self.k*hankel2(1,self.k*safe)/safe;h[diagonal]=0
            weight=weights[i,:,None]*weights[j,None,:]
            bs=p.T@(green*weight)@p
            bkp=p.T@(h*np.einsum('ijc,c->ij',difference,g.normals[i])*weight)@p
            s,kp=_sk_blocks_near_linear(g.elements[i],g.elements[j],self.k,True,8,8)
            append(i,j,s,kp,bs,bkp)
            if i!=j:
                _,kp_reverse=_sk_blocks_near_linear(g.elements[j],g.elements[i],self.k,True,8,8,False,True)
                bkp_reverse=p.T@(-h.T*np.einsum('ijc,c->ij',difference.transpose(1,0,2),g.normals[j])*weight.T)@p
                append(j,i,s.T,kp_reverse,bs.T,bkp_reverse)
        for kind in delta:
            self.correction[kind]=coo_matrix((delta[kind],(rr,cc)),shape=(2*self.m,2*self.m)).tocsr()
            self.near[kind]=coo_matrix((exact[kind],(rr,cc)),shape=(2*self.m,2*self.m)).tocsr()
        self.correction['K']=self.correction['KP'].T.tocsr()
        self.near['K']=self.near['KP'].T.tocsr()

    def _project(self, y, coefficient, derivative=False):
        y=y.reshape(self.m,self.order,-1)*self.weights.reshape(self.m,self.order,1)
        if derivative:
            out=y.sum(axis=1)[:,None,:]*np.array([-1.,1.])[None,:,None]/self.geometry.lengths[:,None,None]
        else: out=np.einsum('qa,eqr->ear',self.phi,y)
        if coefficient is not None:out*=np.asarray(coefficient)[:,None,None]
        return self.P.T@out.reshape(2*self.m,-1)

    def apply(self, kind, x, source_mask=None, coefficient=None):
        self.checkpoint();self.calls+=1
        x=np.asarray(x,complex);vector=x.ndim==1
        if vector:x=x[:,None]
        local=(self.P@x).reshape(self.m,2,-1)
        if source_mask is not None:local*=np.asarray(source_mask)[:,None,None]
        density=np.einsum('qa,ear->eqr',self.phi,local).reshape(self.m*self.order,-1)
        strengths=self.weights[:,None]*density
        if kind=='W':
            derivative=(local[:,1]-local[:,0])/self.geometry.lengths[:,None]
            d=np.repeat(derivative,self.order,axis=0)*self.weights[:,None]
            strength=np.column_stack((d,strengths*self.normals[:,0,None],strengths*self.normals[:,1,None]))
            value,_=evaluate(self.points,self.k,strength,eps=self.eps,threads=self.threads,plan=self.native_plan)
            d,nx,ny=np.split(value,3,axis=1)
            y=self._project(d,coefficient,True)-self.k**2*self._project(nx*self.normals[:,0,None]+ny*self.normals[:,1,None],coefficient)
        else:
            value,gradient=evaluate(self.points,self.k,
                charges=None if kind=='K' else strengths,
                dipoles=strengths if kind=='K' else None,normals=self.normals,
                gradient=kind=='KP',eps=self.eps,threads=self.threads,plan=self.native_plan)
            if kind=='KP':value=np.einsum('qcr,qc->qr',gradient,self.normals)
            y=self._project(value,coefficient)
        corr=self.correction[kind]@local.reshape(2*self.m,-1)
        if coefficient is not None:corr*=np.repeat(coefficient,2)[:,None]
        y+=self.P.T@corr
        return y[:,0] if vector else y

    def apply_many(self,requests):
        """Share a native tree traversal among material routes and S/K' loads."""
        if len(requests)==1:return [self.apply(*requests[0])]
        self.checkpoint();self.calls+=1
        groups={};routes=[];charges=[];dipoles=[];start=0
        want_gradient=any(r[0]=='KP' for r in requests)
        has_dipole=any(r[0]=='K' for r in requests)
        for kind,x,mask,coefficient in requests:
            x=np.asarray(x,complex);vector=x.ndim==1
            if vector:x=x[:,None]
            local=(self.P@x).reshape(self.m,2,-1)
            if mask is not None:local*=np.asarray(mask)[:,None,None]
            key=('S' if kind=='KP' else kind,local.shape)
            candidates=groups.setdefault(key,[])
            group=next((span for other,span in candidates if np.array_equal(other,local)),None)
            if group is None:
                density=np.einsum('qa,ear->eqr',self.phi,local).reshape(len(self.points),-1)
                strength=self.weights[:,None]*density
                if kind=='W':
                    derivative=(local[:,1]-local[:,0])/self.geometry.lengths[:,None]
                    d=np.repeat(derivative,self.order,axis=0)*self.weights[:,None]
                    strength=np.column_stack((d,strength*self.normals[:,0,None],strength*self.normals[:,1,None]))
                count=strength.shape[1]
                charges.append(np.zeros_like(strength) if kind=='K' else strength)
                dipoles.append(strength if kind=='K' else np.zeros_like(strength))
                group=(start,start+count);candidates.append((local,group));start+=count
            routes.append((kind,local,coefficient,group,vector))
        value,gradient=evaluate(self.points,self.k,np.column_stack(charges),
            dipoles=np.column_stack(dipoles) if has_dipole else None,normals=self.normals,
            gradient=want_gradient,eps=self.eps,threads=self.threads,plan=self.native_plan)
        results=[]
        for kind,local,coefficient,(first,last),vector in routes:
            v=value[:,first:last]
            if kind=='KP':v=np.einsum('qcr,qc->qr',gradient[:,:,first:last],self.normals)
            if kind=='W':
                d,nx,ny=np.split(v,3,axis=1)
                y=self._project(d,coefficient,True)-self.k**2*self._project(nx*self.normals[:,0,None]+ny*self.normals[:,1,None],coefficient)
            else:y=self._project(v,coefficient)
            correction=self.correction[kind]@local.reshape(2*self.m,-1)
            if coefficient is not None:correction*=np.repeat(coefficient,2)[:,None]
            y+=self.P.T@correction
            results.append(y[:,0] if vector else y)
        return results

    def sparse(self, kind, source_mask=None, coefficient=None):
        a=self.near[kind]
        if source_mask is not None:a=a.multiply(np.repeat(source_mask,2)[None,:])
        if coefficient is not None:a=a.multiply(np.repeat(coefficient,2)[:,None])
        return (self.P.T@a@self.P).tocsr()

    @property
    def storage_bytes(self):
        arrays=(self.points,self.weights,self.normals,self.ids)
        matrices=list(self.near.values())+list(self.correction.values())+[self.P]
        return sum(a.nbytes for a in arrays)+sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes for a in matrices)
