"""Experimental one-pass tile assembly, compressed storage and reusable inverse.

Uses the production material/quadrature equations through PreparedOracle.
No full system matrix is allocated during assembly; each tile is finalized once.
The experiment has an explicit retained-storage cap and no dense fallback.
"""
from prepared_oracle import *
import scipy.linalg as la


class StreamedOperator:
    def __init__(self,oracle,coordinates,tile=128,tolerance=1e-14,budget=512*1024**2,checkpoint=None):
        self.n=oracle.n;self.bytes=0;self.tiles={};self.checkpoint=checkpoint or (lambda:None)
        self.entries=self.calls=self.max_entries=0
        self.row_error=np.zeros(self.n);self.row_norm=np.zeros(self.n)
        self.groups=[]
        def divide(ids):
            if len(ids)<=tile:self.groups.append(ids);return
            axis=int(np.argmax(np.ptp(coordinates[ids],axis=0)))
            ids=ids[np.argsort(coordinates[ids,axis],kind='mergesort')]
            divide(ids[:len(ids)//2]);divide(ids[len(ids)//2:])
        divide(np.arange(self.n))
        # Break the recursive function's closure cycle so this operator and
        # its tile storage can be released immediately after inverse building.
        divide=None
        self.group_id=np.empty(self.n,int);self.local_id=np.empty(self.n,int)
        for i,ids in enumerate(self.groups):self.group_id[ids]=i;self.local_id[ids]=np.arange(len(ids))
        compressed=0;peak_tile=0
        for i,rows in enumerate(self.groups):
            for j,cols in enumerate(self.groups):
                self.checkpoint()
                raw,tail=oracle.get_with_error(rows,cols)
                peak_tile=max(peak_tile,raw.nbytes+tail.nbytes)
                payload=(raw,None)
                if i!=j and np.any(raw):
                    u,s,v=la.svd(raw,full_matrices=False,check_finite=False)
                    energy=np.sqrt(np.cumsum(s[::-1]**2)[::-1])
                    rank=int(np.count_nonzero(energy>tolerance*max(np.linalg.norm(s),1e-300)))
                    if 16*rank*(len(rows)+len(cols)) < raw.nbytes:
                        left=u[:,:rank].copy();right=(s[:rank,None]*v[:rank]).copy()
                        reconstructed=left@right
                        tail+=abs(raw-reconstructed)
                        raw=reconstructed;payload=(left,right);compressed+=1
                elif i!=j:
                    payload=(np.empty((len(rows),0),complex),np.empty((0,len(cols)),complex));compressed+=1
                self.row_norm[rows]+=np.sum(abs(raw),axis=1)
                self.row_error[rows]+=np.sum(tail,axis=1)
                self.bytes+=sum(a.nbytes for a in payload if a is not None)
                if self.bytes>budget:raise MemoryError('Compressed operator exceeded its retained-storage cap.')
                self.tiles[i,j]=payload
        self.evidence=dict(unknowns=self.n,tiles=len(self.tiles),compressed_tiles=compressed,
            retained_bytes=self.bytes,max_query_bytes=peak_tile,geometry_queries=oracle.calls,
            geometry_coefficients=oracle.entries,dropped_routes=oracle.dropped_routes,
            row_error_bound=float(self.row_error.max()),storage_budget=budget)

    def get(self,rows,cols):
        rows,cols=np.asarray(rows,int),np.asarray(cols,int)
        self.entries+=len(rows)*len(cols);self.calls+=1;self.max_entries=max(self.max_entries,len(rows)*len(cols))
        result=np.empty((len(rows),len(cols)),complex)
        for i in np.unique(self.group_id[rows]):
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
