"""Experimental HODLR operator AND inverse; no dense A retained after build.

Current construction uses a dense-file coefficient oracle. Its timing is NOT
matrix-free electromagnetic assembly performance. Oracle accounting records
all coefficient accesses, including the expensive full-block error checks.
"""
from common import *
import scipy.linalg as la
import hierarchical_factor as hf

class Oracle:
    def __init__(self,a):
        self.a=a; self.n=len(a); self.entries=0; self.calls=0; self.max_entries=0
    def get(self,rows,cols):
        entries=len(rows)*len(cols)
        self.entries+=entries; self.calls+=1; self.max_entries=max(self.max_entries,entries)
        return np.array(self.a[np.ix_(rows,cols)],copy=True)

class Block(hf.Block):
    def __init__(self,oracle,rows,cols):
        self.oracle,self.rows,self.cols=oracle,rows,cols
        self.shape=len(rows),len(cols);self.checkpoint=lambda: None
    def row(self,i):return self.oracle.get(self.rows[i:i+1],self.cols)[0]
    def col(self,j):return self.oracle.get(self.rows,self.cols[j:j+1])[:,0]
    def error(self,u,v):
        total=error=largest=0.;pivot=0
        for start in range(0,len(self.rows),32):
            original=self.oracle.get(self.rows[start:start+32],self.cols)
            total+=float(np.vdot(original,original).real)
            original-=u[start:start+32] @ v
            norms=np.sum(abs(original)**2,axis=1);error+=float(norms.sum())
            if len(norms) and norms.max()>largest:
                largest=float(norms.max());pivot=start+int(np.argmax(norms))
        return np.sqrt(error/max(total,1e-300)),pivot

class Node(hf.Node):
    def matmul(self,b,trans=0):
        def op(a):return a if trans==0 else a.T if trans==1 else a.conj().T
        if self.leaf:return op(self.raw) @ b
        n=self.left.n
        x,y=b[:n],b[n:]
        first=self.left.matmul(x,trans);second=self.right.matmul(y,trans)
        if trans==0:
            first+=self.u12 @ (self.v12 @ y);second+=self.u21 @ (self.v21 @ x)
        else:
            first+=op(self.v21) @ (op(self.u21) @ y)
            second+=op(self.v12) @ (op(self.u12) @ x)
        return np.vstack((first,second))

class CompressedSystem:
    def __init__(self,oracle,coordinates,tolerance=1e-12,leaf=128):
        self.tolerance=tolerance;self.leaf=leaf
        self.bytes=0;self.ranks=[];self.leaves=0
        self.row_norm=np.zeros(oracle.n);self.row_error=np.zeros(oracle.n)
        self.column_norm=np.zeros(oracle.n);self.column_error=np.zeros(oracle.n)
        def ordering(ids):
            if len(ids)<=leaf:return ids
            axis=int(np.argmax(np.ptp(coordinates[ids],axis=0)))
            ids=ids[np.argsort(coordinates[ids,axis],kind='mergesort')]
            mid=len(ids)//2
            return np.r_[ordering(ids[:mid]),ordering(ids[mid:])]
        self.permutation=ordering(np.arange(oracle.n))
        self.root=self.build(oracle,self.permutation)
        self.evidence=dict(bytes=self.bytes,max_rank=max(self.ranks or [0]),leaves=self.leaves,
            accesses=oracle.entries,oracle_calls=oracle.calls,max_query_entries=oracle.max_entries,
            relative_inf_error_bound=float(self.row_error.max()/self.row_norm.max()),
            relative_one_error_bound=float(self.column_error.max()/self.column_norm.max()))
        # The oracle is deliberately not assigned to self or any node.
    def factor(self,a):
        self.bytes+=a.nbytes+8*len(a)
        return la.lu_factor(np.array(a,order='F'),overwrite_a=True,check_finite=False)
    def inspect_block(self,oracle,rows,cols,u,v):
        for start in range(0,len(rows),32):
            rr=rows[start:start+32]
            original=oracle.get(rr,cols)
            self.row_norm[rr]+=np.sum(abs(original),axis=1)
            self.column_norm[cols]+=np.sum(abs(original),axis=0)
            original-=u[start:start+32] @ v
            self.row_error[rr]+=np.sum(abs(original),axis=1)
            self.column_error[cols]+=np.sum(abs(original),axis=0)
    def build(self,oracle,ids):
        node=Node();node.n=len(ids);node.leaf=len(ids)<=self.leaf
        if not node.leaf:
            mid=len(ids)//2;left,right=ids[:mid],ids[mid:]
            try:
                node.u12,node.v12,_=hf.compress(Block(oracle,left,right),tolerance=self.tolerance)
                node.u21,node.v21,_=hf.compress(Block(oracle,right,left),tolerance=self.tolerance)
            except hf.HierarchicalRejected:
                if len(ids)>512:raise
                node.leaf=True
                for name in ('u12','u21','v12','v21'):
                    if hasattr(node,name):delattr(node,name)
        if node.leaf:
            node.raw=oracle.get(ids,ids)
            self.row_norm[ids]+=np.sum(abs(node.raw),axis=1)
            self.column_norm[ids]+=np.sum(abs(node.raw),axis=0)
            self.bytes+=node.raw.nbytes;self.leaves+=1
            node.lu=self.factor(node.raw)
            return node
        self.inspect_block(oracle,left,right,node.u12,node.v12)
        self.inspect_block(oracle,right,left,node.u21,node.v21)
        node.left,node.right=self.build(oracle,left),self.build(oracle,right)
        node.e1,node.e2=node.left.solve(node.u12),node.right.solve(node.u21)
        r,s=node.u12.shape[1],node.u21.shape[1];self.ranks.extend((r,s))
        self.bytes+=sum(v.nbytes for v in (node.u12,node.v12,node.u21,node.v21,node.e1,node.e2))
        small=np.eye(r+s,dtype=complex)
        small[:r,r:]=node.v12 @ node.e2;small[r:,:r]=node.v21 @ node.e1
        node.lu=self.factor(small) if r+s else None
        return node
    def apply(self,b,solve=False,trans=0):
        b=np.asarray(b);vector=b.ndim==1
        if vector:b=b[:,None]
        ordered=(self.root.solve if solve else self.root.matmul)(b[self.permutation],trans)
        value=np.empty_like(ordered);value[self.permutation]=ordered
        return value[:,0] if vector else value
