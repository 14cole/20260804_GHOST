"""Compress dimensionally balanced equations rather than mixed trace/flux units."""
from common import *
from compressed_system import CompressedSystem,Oracle

class ScaledOracle(Oracle):
    def __init__(self,a,row,col):
        super().__init__(a);self.row=row;self.col=col
    def get(self,rows,cols):
        value=super().get(rows,cols)
        value/=self.row[rows,None];value/=self.col[None,cols]
        return value

class EquilibratedSystem:
    def __init__(self,a,xy,tolerance=1e-12):
        self.row,self.col,_=rcs._equilibrated_scaling_and_norm_1(a)
        self.inner=CompressedSystem(ScaledOracle(a,self.row,self.col),xy,tolerance)
        self.bytes=self.inner.bytes+self.row.nbytes+self.col.nbytes
        self.row_error=self.row*self.col.max()*self.inner.row_error
        self.evidence=dict(self.inner.evidence,bytes=self.bytes,
            note='Error/norm and access evidence refer to scaled A; scaling uses two additional passes.')
    def apply(self,b,solve=False,trans=0):
        b=np.asarray(b);vector=b.ndim==1
        if vector:b=b[:,None]
        r,c=(self.row,self.col) if trans==0 else (self.col,self.row)
        if solve:value=self.inner.apply(b/r[:,None],solve=True,trans=trans)/c[:,None]
        else:value=r[:,None]*self.inner.apply(c[:,None]*b,trans=trans)
        return value[:,0] if vector else value
