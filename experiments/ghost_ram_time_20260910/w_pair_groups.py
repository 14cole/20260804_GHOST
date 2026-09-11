"""Grade regular W pairs independently; retain original singular integration."""
from common import *
import inspect
from assembly_variants import clone

def pairs(d_mat,i0,j0,batch_ij,batch_ji,ratio,mirrored,lengths,node_ids,p0,seg,normals,k,real_k,fg,order,lock):
    active=batch_ij | (batch_ji if mirrored else False)
    # Existing regular pairs below three lengths keep the original 16-point rule.
    groups=((active & (ratio<3),max(16,order)),(active & (ratio>=3),max(8,order)))
    for mask,q in groups:
        ii,jj=np.nonzero(mask)
        if not len(ii):continue
        oo,ss=ii+i0,jj+j0
        qt,qw=ops._get_quadrature(q)
        acc=np.zeros((4,len(ii)),complex)
        gb=np.empty(len(ii),complex);kr=np.empty(len(ii));work=np.empty(len(ii))
        for t,w in zip(qt,qw):
            ro=p0[oo]+t*seg[oo]
            for u,v in zip(qt,qw):
                rs=p0[ss]+u*seg[ss]
                dist=np.maximum(np.linalg.norm(ro-rs,axis=1),ops.EPS)
                if real_k:ops._far_kernel_argument(k,dist,kr)
                fg(k,real_k,dist,kr,work,gb)
                acc+=np.array([(1-t)*(1-u),(1-t)*u,t*(1-u),t*u])[:,None]*(w*v*gb)[None,:]
        lp=lengths[oo]*lengths[ss]
        acc*=lp[None,:]
        blocksum=acc.sum(axis=0)/np.maximum(lp,ops.EPS**2)
        factor=-complex(k)**2*np.sum(normals[oo]*normals[ss],axis=1)
        with lock:
            for transpose in (False,True) if mirrored else (False,):
                keep=(batch_ji if transpose else batch_ij)[ii,jj]
                rows=node_ids[ss if transpose else oo];cols=node_ids[oo if transpose else ss]
                for a in range(2):
                    for b in range(2):
                        value=factor*acc[2*b+a if transpose else 2*a+b]+ops._TANGENT_OUTER[a,b]*blocksum
                        np.add.at(d_mat,(rows[keep,a],cols[keep,b]),value[keep])

def candidate():
    original=ops._assemble_linear_hypersingular_matrix
    source=inspect.getsource(inspect.unwrap(original))
    start=source.index('            src_pts = quad_pts[src_slice]')
    stop=source.index('\n        if local_scalar:',start)
    new='''            audit_pairs(d_mat,i0,j0,batch_ij,batch_ji,centre_dist/scale,mirrored,
                lengths,node_ids,p0_arr,seg_arr,normals_arr,k0,real_k,far_green,
                max(int(obs_order),int(src_order)),write_lock)
'''
    fn=clone(original,[(source[start:stop],new)])
    fn.__globals__['audit_pairs']=pairs
    return fn
