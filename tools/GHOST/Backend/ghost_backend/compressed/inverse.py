"""Experimental capped HODLR operator/inverse with checked physical solves."""
import numpy as np
import scipy.linalg as la
import ghost_backend.linalg.hierarchical as hf
import warnings

class Oracle:
    def __init__(self,a):
        self.a=a; self.n=len(a); self.entries=0; self.calls=0; self.max_entries=0
    def get(self,rows,cols):
        entries=len(rows)*len(cols)
        self.entries+=entries; self.calls+=1; self.max_entries=max(self.max_entries,entries)
        return np.array(self.a[np.ix_(rows,cols)],copy=True)


class PreparedAccess:
    """Internal tile access: tree indices are already valid unique subsets."""
    def __init__(self,oracle):
        self.get=oracle._get


class Block(hf.Block):
    def __init__(self,oracle,rows,cols,checkpoint):
        self.oracle,self.rows,self.cols=oracle,rows,cols
        self.shape=len(rows),len(cols);self.checkpoint=checkpoint
    def row(self,i):return self.oracle.get(self.rows[i:i+1],self.cols)[0]
    def col(self,j):return self.oracle.get(self.rows,self.cols[j:j+1])[:,0]
    def error(self,u,v):
        total=error=largest=0.;pivot=0
        width=max(1,min(32,(16*1024**2)//max(16*len(self.cols),1)))
        for start in range(0,len(self.rows),width):
            self.checkpoint()
            original=self.oracle.get(self.rows[start:start+width],self.cols)
            total+=float(np.vdot(original,original).real)
            original-=u[start:start+width] @ v
            norms=np.sum(abs(original)**2,axis=1);error+=float(norms.sum())
            if len(norms) and norms.max()>largest:
                largest=float(norms.max());pivot=start+int(np.argmax(norms))
        return np.sqrt(error/max(total,1e-300)),pivot

class Node(hf.Node):
    def solve(self,b,trans=0):
        self.checkpoint()
        return super().solve(b,trans)
    def matmul(self,b,trans=0):
        self.checkpoint()
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
    def __init__(self,oracle,coordinates,tolerance=1e-12,leaf=128,budget=512*1024**2,checkpoint=None,inverse_only=False):
        self.inverse_only=bool(inverse_only)
        if not isinstance(leaf,(int,np.integer)) or not 1<=leaf<=512:
            raise ValueError('Leaf size must be in 1..512.')
        if not np.isfinite(tolerance) or not 0<tolerance<1 or not np.isfinite(budget) or budget<=0:
            raise ValueError('Invalid tolerance or storage budget.')
        coordinates=np.asarray(coordinates,float)
        if oracle.n<=0 or coordinates.ndim!=2 or coordinates.shape[0]!=oracle.n or not coordinates.shape[1] or not np.all(np.isfinite(coordinates)):
            raise ValueError('Invalid coordinates.')
        self.n=oracle.n;self.budget=budget;self.checkpoint=checkpoint or (lambda:None)
        self.checkpoint()
        self.tolerance=tolerance;self.leaf=leaf
        self.bytes=0;self.ranks=[];self.leaves=0
        self.row_norm=np.zeros(oracle.n);self.row_error=np.zeros(oracle.n)
        self.column_norm=np.zeros(oracle.n);self.column_error=np.zeros(oracle.n)
        self.permutation=hf.spatial_order(coordinates,np.arange(oracle.n),leaf)
        self.reserve(sum(a.nbytes for a in (self.row_norm,self.row_error,self.column_norm,self.column_error,self.permutation)))


        access=PreparedAccess(oracle) if hasattr(oracle,'_get') else oracle
        self.root=self.build(access,self.permutation)
        self.evidence=dict(bytes=self.bytes,max_rank=max(self.ranks or [0]),leaves=self.leaves,
            accesses=oracle.entries,oracle_calls=oracle.calls,max_query_entries=oracle.max_entries,
            relative_inf_error_bound=float(self.row_error.max()/max(self.row_norm.max(),1e-300)),
            relative_one_error_bound=float(self.column_error.max()/max(self.column_norm.max(),1e-300)),
            storage_budget=budget,inverse_only=self.inverse_only)
        if self.inverse_only:
            self.evidence['relative_inf_error_bound']=None
            self.evidence['relative_one_error_bound']=None

    def reserve(self,count):
        if self.bytes+count>self.budget:
            raise MemoryError('Compressed inverse exceeded its retained-storage cap.')
        self.bytes+=count
    def factor(self,a):
        self.checkpoint()
        self.reserve(a.nbytes+8*len(a))
        if not np.all(np.isfinite(a)):
            raise ValueError('Nonfinite compressed factor.')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            return la.lu_factor(np.array(a,order='F'),overwrite_a=True,check_finite=False)
    def inspect_block(self,oracle,rows,cols,u,v):
        width=max(1,min(32,(16*1024**2)//max(16*len(cols),1)))
        for start in range(0,len(rows),width):
            self.checkpoint()
            rr=rows[start:start+width]
            original=oracle.get(rr,cols)
            self.row_norm[rr]+=np.sum(abs(original),axis=1)
            self.column_norm[cols]+=np.sum(abs(original),axis=0)
            original-=u[start:start+width] @ v
            self.row_error[rr]+=np.sum(abs(original),axis=1)
            self.column_error[cols]+=np.sum(abs(original),axis=0)
    def build(self,oracle,ids):
        self.checkpoint()
        node=Node();node.n=len(ids);node.leaf=len(ids)<=self.leaf;node.checkpoint=self.checkpoint
        if not node.leaf:
            mid=len(ids)//2;left,right=ids[:mid],ids[mid:]
            try:
                node.u12,node.v12,_=hf.compress(Block(oracle,left,right,self.checkpoint),tolerance=self.tolerance)
                node.u21,node.v21,_=hf.compress(Block(oracle,right,left,self.checkpoint),tolerance=self.tolerance)
            except hf.HierarchicalRejected:
                if len(ids)>512:raise
                node.leaf=True
                for name in ('u12','u21','v12','v21'):
                    if hasattr(node,name):delattr(node,name)
        if node.leaf:
            if self.inverse_only:
                self.leaves+=1
                node.lu=self.factor(oracle.get(ids,ids))
                return node
            self.reserve(len(ids)**2*16)
            node.raw=oracle.get(ids,ids)
            self.row_norm[ids]+=np.sum(abs(node.raw),axis=1)
            self.column_norm[ids]+=np.sum(abs(node.raw),axis=0)
            self.leaves+=1
            node.lu=self.factor(node.raw)
            return node
        self.reserve(sum(v.nbytes for v in (node.u12,node.v12,node.u21,node.v21)))
        if not self.inverse_only:
            self.inspect_block(oracle,left,right,node.u12,node.v12)
            self.inspect_block(oracle,right,left,node.u21,node.v21)
        node.left,node.right=self.build(oracle,left),self.build(oracle,right)
        self.reserve(node.u12.nbytes+node.u21.nbytes)
        node.e1,node.e2=node.left.solve(node.u12),node.right.solve(node.u21)
        r,s=node.u12.shape[1],node.u21.shape[1];self.ranks.extend((r,s))
        small=np.eye(r+s,dtype=complex)
        small[:r,r:]=node.v12 @ node.e2;small[r:,:r]=node.v21 @ node.e1
        node.lu=self.factor(small) if r+s else None
        if self.inverse_only:
            self.bytes-=node.u12.nbytes+node.u21.nbytes
            del node.u12,node.u21
        return node
    def apply(self,b,solve=False,trans=0):
        if self.inverse_only and not solve:raise ValueError('Inverse-only preconditioner has no retained operator.')
        if trans not in (0,1,2):raise ValueError('Invalid transpose mode.')
        b=np.asarray(b);vector=b.ndim==1
        if vector:b=b[:,None]
        if b.ndim!=2 or b.shape[0]!=self.n or not b.shape[1] or not np.all(np.isfinite(b)):
            raise ValueError('Invalid compressed RHS.')
        ordered=(self.root.solve if solve else self.root.matmul)(b[self.permutation],trans)
        value=np.empty_like(ordered);value[self.permutation]=ordered
        return value[:,0] if vector else value

    def solve_checked(self,b,input_error=None,trans=0,initial=None,limit=1e-12,overwrite_initial=False):
        """Refine/reject against the stored operator plus coefficient error.

        input_error contains row (column for transpose) sums of the preceding
        tile/pruning error. It must be provided whenever that oracle is approximate.
        """
        if self.inverse_only:raise ValueError('Preconditioner must be checked against a separate accurate operator.')
        b=np.asarray(b,complex);vector=b.ndim==1
        if vector:b=b[:,None]
        if b.ndim!=2 or b.shape[0]!=self.n or not b.shape[1] or not np.all(np.isfinite(b)):
            raise ValueError('Invalid compressed RHS.')
        if trans not in (0,1,2) or not np.isfinite(limit) or not 0<limit<1:
            raise ValueError('Invalid transpose or acceptance limit.')
        x=(self.apply(b,solve=True,trans=trans) if initial is None else
           np.asarray(initial,complex) if overwrite_initial else np.array(initial,complex,copy=True))
        if x.ndim==1:x=x[:,None]
        if x.shape!=b.shape or not np.all(np.isfinite(x)):
            raise ValueError('Invalid initial solution or RHS.')
        if not x.flags.writeable:raise ValueError('Initial solution must be writable when adopted.')
        error=self.row_error if trans==0 else self.column_error
        norm=self.row_norm if trans==0 else self.column_norm
        source=np.zeros(self.n) if input_error is None else np.asarray(input_error,float)
        if source.shape!=(self.n,) or not np.all(np.isfinite(source)) or np.any(source<0):
            raise ValueError('Invalid incoming coefficient error.')
        total=float(np.max(error+source));lower=max(float(np.max(norm-source)),0.)
        for step in range(8):
            residual=b-self.apply(x,trans=trans)
            den=np.maximum(lower*np.max(abs(x),axis=0)+np.max(abs(b),axis=0),1e-300)
            bounds=(np.max(abs(residual),axis=0)+total*np.max(abs(x),axis=0))/den
            if np.all(np.isfinite(bounds)) and np.max(bounds)<=limit:
                self.evidence['last_original_residual_bound']=float(np.max(bounds))
                self.evidence['last_refinements']=step
                return x[:,0] if vector else x
            x+=self.apply(residual,solve=True,trans=trans)
        raise hf.HierarchicalRejected('Compressed solve could not satisfy the original-operator error bound.')
