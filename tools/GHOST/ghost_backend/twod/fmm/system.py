"""Sparse equation routing for matrix-free material systems."""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator


def mapping(ids, destinations, shape, weights=1.):
    return coo_matrix((np.broadcast_to(weights,len(ids)),(destinations,ids)),shape=shape,dtype=complex).tocsr()


class FMMSystem(LinearOperator):
    def __init__(self,n,jumps=None):
        super().__init__(dtype=np.dtype(complex),shape=(n,n))
        self.jumps=csr_matrix((n,n),dtype=complex) if jumps is None else jumps.astype(complex).tocsr()
        self.terms=[];self.endpoints=np.empty(0,int)
        self.spatial_coarse_eligible=False

    def __len__(self):return self.shape[0]

    def add_combined_field(self,kernel,eta):
        """Closed PEC route; retain separate terms for near ILU and inspection."""
        if self.terms or len(self)!=kernel.n:
            raise ValueError('Combined field requires an unmapped, single-kernel system.')
        self.combined_field_eta=complex(eta)
        self.add(kernel,'K')
        self.add(kernel,'S',weight=eta)
        self._combined_kernel=kernel

    def add(self,kernel,kind,rows=None,cols=None,weight=1.,mask=None,coefficient=None):
        if hasattr(self,'_combined_kernel'):
            raise ValueError('A fused PEC system cannot accept additional material routes.')
        nn=kernel.n;ids=np.arange(nn)
        r=ids if rows is None else np.asarray(rows)
        c=ids if cols is None else np.asarray(cols)
        ri=np.flatnonzero(r>=0);ci=np.flatnonzero(c>=0)
        weights=np.broadcast_to(weight,nn)[ri]
        R=mapping(ri,r[ri],(len(self),nn),weights)
        C=mapping(c[ci],ci,(nn,len(self)))
        self.terms.append((kernel,kind,R,C,mask,coefficient))

    def _apply(self,x,adjoint=False):
        vector=x.ndim==1
        if vector:x=x[:,None]
        if adjoint:
            original=x
            x=x.copy();x[self.endpoints]=0
            result=self.jumps.conj().T@x
        else:result=self.jumps@x
        if hasattr(self,'_combined_kernel'):
            result+=self._combined_kernel.apply_combined(x,self.combined_field_eta,adjoint)
            if adjoint:result[self.endpoints]+=original[self.endpoints]
            else:result[self.endpoints]=x[self.endpoints]
            return result[:,0] if vector else result
        groups={}
        for kernel,kind,R,C,mask,coefficient in self.terms:
            if adjoint:
                transposed={'S':'S','KP':'K','K':'KP','W':'W'}[kind]
                request=(transposed,R.T@x.conj(),coefficient,mask)
                destination=C.conj().T
            else:request=(kind,C@x,mask,coefficient);destination=R
            groups.setdefault(kernel,[]).append((request,destination))
        for kernel,routes in groups.items():
            values=kernel.apply_many([r[0] for r in routes])
            for value,(_,destination) in zip(values,routes):
                result+=destination@(value.conj() if adjoint else value)
        if adjoint:result[self.endpoints]+=original[self.endpoints]
        else:result[self.endpoints]=x[self.endpoints]
        return result[:,0] if vector else result

    def _matvec(self,x):return self._apply(x)
    def _matmat(self,x):return self._apply(x)
    def _rmatvec(self,x):return self._apply(x,True)
    def _rmatmat(self,x):return self._apply(x,True)

    def sparse_near(self):
        result=self.jumps.copy()
        for kernel,kind,R,C,mask,coefficient in self.terms:
            result+=R@kernel.sparse(kind,mask,coefficient)@C
        if len(self.endpoints):
            result=result.tolil()
            for i in self.endpoints:result.rows[i]=[i];result.data[i]=[1.]
        return result.tocsc()

    @property
    def kernels(self):return list({id(t[0]):t[0] for t in self.terms}.values())

    def report(self):
        return dict(method='galerkin_fmm_gmres',unknowns=len(self),
            representation='combined_field_pec' if hasattr(self,'combined_field_eta') else 'original_material_equations',
            quadrature_points=sum(len(k.points) for k in self.kernels),
            near_pairs=sum(len(k.pairs) for k in self.kernels),
            operator_storage_bytes=sum(k.storage_bytes for k in self.kernels),
            kernel_tolerances=[k.eps for k in self.kernels],
            quadrature_orders=[k.order for k in self.kernels],
            quadrature_policies=[k.quadrature_policy for k in self.kernels],
            combined_field_fused=hasattr(self,'_combined_kernel'),
            dense_matrix_built=False,approximation_error_certified=False,
            spatial_coarse_eligible=self.spatial_coarse_eligible,
            residual_operator='locally_corrected_fmm_galerkin')
