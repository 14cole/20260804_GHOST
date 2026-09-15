"""Polynomial Galerkin operators: point FMM plus accurate near corrections.

Corrections live on panel basis incidences, preserving discontinuous material
weights and source masks even at shared nodes. No dense global operator is built.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.special import hankel2
from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
from ghost_backend.twod.fmm.kernel import evaluate, NativePlan
from ghost_backend.twod.basis import values, derivative_matrix


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
                if count*576*geometry.node_ids.shape[1]**2 > budget:
                    raise MemoryError('FMM near interactions exceed the configured storage budget.')
                if not count_only:pairs.extend((i,int(j)) for j in js)
    return count if count_only else pairs


class GalerkinKernel:
    def __init__(self, mesh, k, order=8, eps=1e-10, threads=1,
                 budget=2*1024**3, checkpoint=None, geometry=None, pairs=None, quadrature=0):
        self.mesh, self.k, self.eps, self.threads = mesh, complex(k), eps, threads
        self.checkpoint = checkpoint or (lambda: None)
        self.geometry = geometry or AssemblyGeometry(mesh)
        g=self.geometry
        if not len(g.lengths) or np.any(g.lengths<=0): raise ValueError('FMM needs nondegenerate panels.')
        self.n=len(mesh.nodes);self.m=len(g.lengths);self.width=g.node_ids.shape[1]
        from ghost_backend.twod.fmm.quadrature import quadrature_order
        minimum=max(8,quadrature) if self.width>2 else quadrature
        self.order,self.quadrature_policy=quadrature_order(k,g.lengths.max(),minimum,order,eps)
        if 64*self.m*self.order>budget or pairs is not None and len(pairs)*576*self.width**2>budget:
            raise MemoryError('FMM geometry and near workspace exceed the configured storage budget.')
        t,w=np.polynomial.legendre.leggauss(self.order);t=(t+1)/2;w=w/2
        self.phi=values(t,self.width-1)
        self.dphi=values(t,self.width-1,True)
        self.points=(g.p0[:,None]+t[None,:,None]*g.segments[:,None]).reshape(-1,2)
        self.native_plan=NativePlan(self.points,self.k,self.eps)
        self.points=self.native_plan.points
        self.weights=(g.lengths[:,None]*w).ravel()
        self.normals=np.repeat(g.normals,self.order,axis=0)
        self.ids=g.node_ids.ravel()
        self.P=coo_matrix((np.ones(self.width*self.m),(np.arange(self.width*self.m),self.ids)),shape=(self.width*self.m,self.n)).tocsr()
        self.pairs=near_pairs(g,budget) if pairs is None else pairs
        self.correction={};self.near={};self.calls=0
        self._build()

    def _build(self):
        from ghost_backend.twod.operators import _sk_blocks_near_linear
        # Numeric buffers avoid a Python object for every sparse scalar/index.
        entries=self.width**2*sum(1 if i==j else 2 for i,j in self.pairs)
        rr=np.empty(entries,np.int64);cc=np.empty(entries,np.int64)
        delta={kind:np.empty(entries,complex) for kind in ('S','KP','W')}
        exact={kind:np.empty(entries,complex) for kind in delta}
        cursor=0
        g=self.geometry;q=self.order;p=self.phi
        derivative=derivative_matrix(self.width-1)
        points=self.points.reshape(self.m,q,2);weights=self.weights.reshape(self.m,q)
        def append(i,j,s,kp,bs,bkp):
            nonlocal cursor
            normal=np.dot(g.normals[i],g.normals[j])
            maue=lambda b: -self.k**2*normal*b+derivative.T@b@derivative/(g.lengths[i]*g.lengths[j])
            sl=slice(cursor,cursor+self.width**2);cursor+=self.width**2
            rr[sl]=np.repeat(self.width*i+np.arange(self.width),self.width)
            cc[sl]=np.tile(self.width*j+np.arange(self.width),self.width)
            for kind,a,b in (('S',s,bs),('KP',kp,bkp),('W',maue(s),maue(bs))):
                delta[kind][sl]=(a-b).ravel();exact[kind][sl]=a.ravel()
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
            self.correction[kind]=coo_matrix((delta[kind],(rr,cc)),shape=(self.width*self.m,self.width*self.m)).tocsr()
            self.near[kind]=coo_matrix((exact[kind],(rr,cc)),shape=(self.width*self.m,self.width*self.m)).tocsr()
            self.correction[kind].eliminate_zeros();self.near[kind].eliminate_zeros()
        # CSC transpose views share the KP data, indices and pointer storage.
        self.correction['K']=self.correction['KP'].T
        self.near['K']=self.near['KP'].T

    def _project(self, y, coefficient, derivative=False):
        y=y.reshape(self.m,self.order,-1)*self.weights.reshape(self.m,self.order,1)
        if derivative:
            out=np.einsum('qa,eqr->ear',self.dphi,y)/self.geometry.lengths[:,None,None]
        else: out=np.einsum('qa,eqr->ear',self.phi,y)
        if coefficient is not None:out*=np.asarray(coefficient)[:,None,None]
        return self.P.T@out.reshape(self.width*self.m,-1)

    def apply(self, kind, x, source_mask=None, coefficient=None):
        x=np.asarray(x,complex);vector=x.ndim==1
        if vector:x=x[:,None]
        width=2 if kind=='W' else 8
        if x.shape[1]>width:
            return np.column_stack([self.apply(kind,x[:,j:j+width],source_mask,coefficient)
                                    for j in range(0,x.shape[1],width)])
        self.checkpoint();self.calls+=1
        local=(self.P@x).reshape(self.m,self.width,-1)
        if source_mask is not None:local*=np.asarray(source_mask)[:,None,None]
        density=np.einsum('qa,ear->eqr',self.phi,local).reshape(self.m*self.order,-1)
        strengths=self.weights[:,None]*density
        if kind=='W':
            derivative=np.einsum('qa,ear->eqr',self.dphi,local)/self.geometry.lengths[:,None,None]
            d=derivative.reshape(len(self.points),-1)*self.weights[:,None]
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
        corr=self.correction[kind]@local.reshape(self.width*self.m,-1)
        if coefficient is not None:corr*=np.repeat(coefficient,self.width)[:,None]
        y+=self.P.T@corr
        return y[:,0] if vector else y

    def apply_combined(self,x,eta,adjoint=False):
        """Apply K+eta*S in one density channel, including its exact adjoint."""
        x=np.asarray(x,complex);vector=x.ndim==1
        if vector:x=x[:,None]
        if x.shape[1]>8:
            return np.column_stack([self.apply_combined(x[:,j:j+8],eta,adjoint)
                                    for j in range(0,x.shape[1],8)])
        self.checkpoint();self.calls+=1
        local=(self.P@(x.conj() if adjoint else x)).reshape(self.m,self.width,-1)
        density=np.einsum('qa,ear->eqr',self.phi,local).reshape(len(self.points),-1)
        strengths=self.weights[:,None]*density
        value,gradient=evaluate(self.points,self.k,
            charges=strengths if adjoint else eta*strengths,
            dipoles=None if adjoint else strengths,normals=self.normals,
            gradient=adjoint,eps=self.eps,threads=self.threads,plan=self.native_plan)
        if adjoint:
            value=eta*value+np.einsum('qcr,qc->qr',gradient,self.normals)
        y=self._project(value,None)
        inc=local.reshape(self.width*self.m,-1)
        y+=self.P.T@(self.correction['KP' if adjoint else 'K']@inc+eta*(self.correction['S']@inc))
        if adjoint:y=y.conj()
        return y[:,0] if vector else y

    def apply_many(self,requests):
        """Share a native tree traversal among material routes and S/K' loads."""
        if len(requests)==1:return [self.apply(*requests[0])]
        if sum(3 if r[0]=='W' else 1 for r in requests)>8:
            results=[];chunk=[];loads=0
            for request in requests:
                cost=3 if request[0]=='W' else 1
                if loads+cost>8:
                    results.extend(self.apply_many(chunk));chunk=[];loads=0
                chunk.append(request);loads+=cost
            return results+self.apply_many(chunk)
        # Bound projected point arrays as well as the native workspace. Final
        # node-space outputs are the only arrays spanning the full RHS batch.
        columns=np.asarray(requests[0][1]).shape
        width=max(1,8//sum(3 if r[0]=='W' else 1 for r in requests))
        if len(columns)==2 and columns[1]>width:
            chunks=[self.apply_many([(kind,x[:,j:j+width],mask,c) for kind,x,mask,c in requests])
                    for j in range(0,columns[1],width)]
            return [np.column_stack([c[i] for c in chunks]) for i in range(len(requests))]
        self.checkpoint();self.calls+=1
        groups={};routes=[];charges=[];dipoles=[];start=0
        want_gradient=any(r[0]=='KP' for r in requests)
        has_dipole=any(r[0]=='K' for r in requests)
        has_charge=any(r[0]!='K' for r in requests)
        for kind,x,mask,coefficient in requests:
            x=np.asarray(x,complex);vector=x.ndim==1
            if vector:x=x[:,None]
            local=(self.P@x).reshape(self.m,self.width,-1)
            if mask is not None:local*=np.asarray(mask)[:,None,None]
            key=('S' if kind=='KP' else kind,local.shape)
            candidates=groups.setdefault(key,[])
            group=next((span for other,span in candidates if np.array_equal(other,local)),None)
            if group is None:
                density=np.einsum('qa,ear->eqr',self.phi,local).reshape(len(self.points),-1)
                strength=self.weights[:,None]*density
                if kind=='W':
                    derivative=np.einsum('qa,ear->eqr',self.dphi,local)/self.geometry.lengths[:,None,None]
                    d=derivative.reshape(len(self.points),-1)*self.weights[:,None]
                    strength=np.column_stack((d,strength*self.normals[:,0,None],strength*self.normals[:,1,None]))
                count=strength.shape[1]
                if has_charge:charges.append(np.zeros_like(strength) if kind=='K' else strength)
                if has_dipole:dipoles.append(strength if kind=='K' else np.zeros_like(strength))
                group=(start,start+count);candidates.append((local,group));start+=count
            routes.append((kind,local,coefficient,group,vector))
        value,gradient=evaluate(self.points,self.k,np.column_stack(charges) if has_charge else None,
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
            correction=self.correction[kind]@local.reshape(self.width*self.m,-1)
            if coefficient is not None:correction*=np.repeat(coefficient,self.width)[:,None]
            y+=self.P.T@correction
            results.append(y[:,0] if vector else y)
        return results

    def sparse(self, kind, source_mask=None, coefficient=None):
        a=self.near[kind]
        if source_mask is not None:a=a.multiply(np.repeat(source_mask,self.width)[None,:])
        if coefficient is not None:a=a.multiply(np.repeat(coefficient,self.width)[:,None])
        return (self.P.T@a@self.P).tocsr()

    @property
    def storage_bytes(self):
        arrays=(self.points,self.weights,self.normals,self.ids)
        matrices=list(self.near.values())+list(self.correction.values())+[self.P]
        buffers={}
        for a in list(arrays)+[b for a in matrices for b in (a.data,a.indices,a.indptr)]:
            while isinstance(a.base,np.ndarray):a=a.base
            buffers[id(a)]=a.nbytes
        return sum(buffers.values())
