"""Experimental one-pass tile assembly, compressed storage and reusable inverse.

Uses the production material/quadrature equations through PreparedOracle.
No full system matrix is allocated during assembly; each tile is finalized once.
The experiment has an explicit retained-storage cap and no dense fallback.
"""
from prepared_oracle import *
import scipy.linalg as la
from sweep_compression import _qr_basis
from hierarchical_factor import spatial_order


def tile_payload(raw, tail, tolerance, method):
    """Local scope releases decomposition work before assembling the next tile."""
    if not np.any(raw):
        return (np.empty((len(raw),0),complex), np.empty((0,raw.shape[1]),complex)), raw, tail, True
    if method == 'qr':
        left, right = _qr_basis(raw, tolerance*max(float(np.linalg.norm(raw)),1e-300))
    else:
        u,s,v=la.svd(raw,full_matrices=False,check_finite=False)
        energy=np.sqrt(np.cumsum(s[::-1]**2)[::-1])
        rank=int(np.count_nonzero(energy>tolerance*max(np.linalg.norm(s),1e-300)))
        left=u[:,:rank].copy();right=(s[:rank,None]*v[:rank]).copy()
    if left.nbytes+right.nbytes >= raw.nbytes:
        return (raw,None), raw, tail, False
    reconstructed=left@right
    difference=abs(raw-reconstructed)
    # Rank estimates choose candidates only. Inspect every coefficient after
    # factor formation and include the actual discrepancy in the error bound.
    if np.linalg.norm(difference)>tolerance*max(float(np.linalg.norm(raw)),1e-300):
        return (raw,None), raw, tail, False
    tail+=difference
    return (left,right), reconstructed, tail, True


class StreamedOperator:
    def __init__(self,oracle,coordinates,tile=128,tolerance=1e-14,budget=512*1024**2,checkpoint=None,compression='qr',assemble=True):
        if not isinstance(tile,(int,np.integer)) or not 1 <= tile <= 1024:
            raise ValueError('Tile size must be an integer in 1..1024.')
        if not np.isfinite(tolerance) or not 0<tolerance<1 or not np.isfinite(budget) or budget<=0:
            raise ValueError('Invalid tolerance or storage budget.')
        if compression not in ('qr','svd'):
            raise ValueError('Compression must be qr or svd.')
        coordinates=np.asarray(coordinates,float)
        if oracle.n<=0 or coordinates.ndim!=2 or coordinates.shape[0]!=oracle.n or not coordinates.shape[1] or not np.all(np.isfinite(coordinates)):
            raise ValueError('Coordinates must be finite and match the nonempty system.')
        self.n=oracle.n;self.bytes=0;self.tiles={};self.checkpoint=checkpoint or (lambda:None)
        self.entries=self.calls=self.max_entries=0
        self.row_error=np.zeros(self.n);self.row_norm=np.zeros(self.n)
        order=spatial_order(coordinates,np.arange(self.n),tile)
        # Mirror the balanced tree's leaf sizes, without a recursive closure.
        pending=[order];self.groups=[]
        while pending:
            ids=pending.pop()
            if len(ids)<=tile:self.groups.append(ids)
            else:pending.extend((ids[len(ids)//2:],ids[:len(ids)//2]))
        self.group_id=np.empty(self.n,int);self.local_id=np.empty(self.n,int)
        for i,ids in enumerate(self.groups):self.group_id[ids]=i;self.local_id[ids]=np.arange(len(ids))
        self.bytes=sum(a.nbytes for a in (order,self.group_id,self.local_id,self.row_error,self.row_norm))
        if self.bytes>budget:raise MemoryError('Compressed operator exceeded its retained-storage cap.')
        self.compressed=0;self.peak_tile=0;self.tolerance=tolerance;self.compression=compression;self.budget=budget
        if not assemble:return
        for i,rows in enumerate(self.groups):
            for j,cols in enumerate(self.groups):
                self.checkpoint()
                raw,tail=oracle.get_with_error(rows,cols)
                self.add_tile(i,j,raw,tail)
                raw=tail=payload=None
        self.finalize(oracle)

    def add_tile(self,i,j,raw,tail):
        self.checkpoint()
        rows,cols=self.groups[i],self.groups[j]
        if (i,j) in self.tiles or raw.shape!=(len(rows),len(cols)) or tail.shape!=raw.shape:
            raise ValueError('Duplicate or incorrectly shaped tile.')
        self.peak_tile=max(self.peak_tile,raw.nbytes+tail.nbytes)
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(tail)) or np.any(tail<0):
            raise ValueError('Oracle returned invalid coefficients or error bounds.')
        payload=(raw,None)
        if i!=j:
            payload,raw,tail,accepted=tile_payload(raw,tail,self.tolerance,self.compression)
            self.compressed+=int(accepted)
        self.row_norm[rows]+=np.sum(abs(raw),axis=1)
        self.row_error[rows]+=np.sum(tail,axis=1)
        self.bytes+=sum(a.nbytes for a in payload if a is not None)
        if self.bytes>self.budget:raise MemoryError('Compressed operator exceeded its retained-storage cap.')
        self.tiles[i,j]=payload

    def finalize(self,oracle):
        if len(self.tiles)!=len(self.groups)**2:raise ValueError('Operator has missing tiles.')
        self.evidence=dict(unknowns=self.n,tiles=len(self.tiles),compressed_tiles=self.compressed,
            retained_bytes=self.bytes,max_query_bytes=self.peak_tile,geometry_queries=oracle.calls,
            geometry_coefficients=oracle.entries,dropped_routes=oracle.dropped_routes,
            row_error_bound=float(self.row_error.max()),storage_budget=self.budget,compression=self.compression)

    def get(self,rows,cols):
        rows,cols=CompactOperator._ids(rows,self.n),CompactOperator._ids(cols,self.n)
        return self._get(rows,cols)

    def _get(self,rows,cols):
        """Internal access for index subsets of a validated spatial permutation."""
        if len(rows)*len(cols)*16>16*1024**2:
            raise MemoryError('Coefficient query exceeds the 16 MiB workspace limit.')
        self.entries+=len(rows)*len(cols);self.calls+=1;self.max_entries=max(self.max_entries,len(rows)*len(cols))
        result=np.empty((len(rows),len(cols)),complex)
        for i in np.unique(self.group_id[rows]):
            self.checkpoint()
            ri=np.flatnonzero(self.group_id[rows]==i);local_r=self.local_id[rows[ri]]
            for j in np.unique(self.group_id[cols]):
                ci=np.flatnonzero(self.group_id[cols]==j);local_c=self.local_id[cols[ci]]
                left,right=self.tiles[i,j]
                value=left[np.ix_(local_r,local_c)] if right is None else left[local_r] @ right[:,local_c]
                result[np.ix_(ri,ci)]=value
        return result

    def matmul(self,b,trans=0):
        if trans not in (0,1,2):raise ValueError('Invalid transpose mode')
        b=np.asarray(b);vector=b.ndim==1
        if vector:b=b[:,None]
        if b.ndim!=2 or b.shape[0]!=self.n or not b.shape[1] or not np.all(np.isfinite(b)):
            raise ValueError('Invalid operator RHS.')
        result=np.zeros_like(b,dtype=complex)
        for (i,j),(left,right) in self.tiles.items():
            self.checkpoint()
            rows,cols=self.groups[i],self.groups[j]
            if trans==0:
                result[rows]+=left@b[cols] if right is None else left@(right@b[cols])
            else:
                def adj(a):return a.T if trans==1 else a.conj().T
                result[cols]+=adj(left)@b[rows] if right is None else adj(right)@(adj(left)@b[rows])
        return result[:,0] if vector else result
