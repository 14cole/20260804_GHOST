"""Tiled P0 Helmholtz coefficients with analytic singularity subtraction.

The logarithmic and 1/r parts follow the small-argument Hankel expansions
https://dlmf.nist.gov/10.8. Only their smooth remainders are quadratured near
panels. Self terms are integrated adaptively and normal principal values vanish
on a straight self panel. No Galerkin testing or nodal mass matrix is used.
"""
from functools import lru_cache
import numpy as np
from scipy.special import hankel2,j0,y0,j1,y1
from scipy.integrate import quad


@lru_cache(64)
def gauss(order):
    t,w=np.polynomial.legendre.leggauss(order)
    return (t+1)/2,w/2


def green(k,r,derivative,potential=True):
    safe=np.maximum(r,1e-300)
    if complex(k).imag==0:
        z=float(complex(k).real)*safe
        g=(y0(z)+1j*j0(z))*.25 if potential else None
        h=-float(complex(k).real)*(y1(z)+1j*j1(z))/(4*safe) if derivative else None
    else:
        g=.25j*hankel2(0,k*safe) if potential else None
        h=-.25j*k*hankel2(1,k*safe)/safe if derivative else None
    return g,h


@lru_cache(8192)
def _self_integral(k,length):
    # A dimensionless interval keeps absolute accuracy consistent with length.
    # The integrand is scalar, so its real and imaginary parts go through quad
    # separately instead of quad_vec, which needs SciPy 1.4. QUADPACK's
    # extrapolation handles the endpoint logarithm at t=0.
    def component(part):
        return quad(lambda t:part(green(k,np.array([length*t]),False)[0][0]),
                    0.,.5,epsabs=2e-13,epsrel=2e-12,limit=200)[0]
    return 2*length*complex(component(lambda z:z.real),component(lambda z:z.imag))


def self_single_layer(k,length):
    # Nominally equal panels differ by a few rounding bits after geometric
    # transforms. Share adaptive integrals, then restore their exact lengths
    # with d/dL integral[-L/2,L/2]G = G(k L/2). The omitted term is O(deltaL^2),
    # far below floating-point precision for this 12-significant-digit bucket.
    canonical=float(format(length,'.12g'))
    value=_self_integral(k,canonical)
    delta=length-canonical
    if delta:value+=delta*green(k,np.array([canonical/2]),False)[0][0]
    return value


def point_pairs(g,k,rows,cols,kinds,order):
    """Uncorrected source-Gauss integral for paired midpoint/source indices."""
    rows=np.asarray(rows);cols=np.asarray(cols)
    t,w=gauss(order)
    diff=g.centers[rows,None]-g.p0[cols,None]-t[None,:,None]*g.segments[cols,None]
    r=np.linalg.norm(diff,axis=-1)
    derivative=bool(set(kinds)-{'S'})
    G,H=green(k,np.where(r==0,1.,r),derivative,'S' in kinds)
    if G is not None:G[r==0]=0
    if H is not None:H[r==0]=0
    weight=g.lengths[cols,None]*w
    result={}
    if 'S' in kinds:result['S']=np.sum(G*weight,axis=1)
    if 'KP' in kinds:
        result['KP']=np.sum(H*np.einsum('pqc,pc->pq',diff,g.normals[rows])*weight,axis=1)
    if 'K' in kinds:
        result['K']=-np.sum(H*np.einsum('pqc,pc->pq',diff,g.normals[cols])*weight,axis=1)
    return result


def near_pairs(g,k,rows,cols,kinds,order=16):
    """Analytic static term plus split-Gauss Helmholtz remainder."""
    rows=np.asarray(rows);cols=np.asarray(cols)
    length=g.lengths[cols];tangent=g.segments[cols]/length[:,None]
    normal=g.normals[cols]
    displacement=g.centers[rows]-g.p0[cols]
    u=np.sum(displacement*tangent,axis=1)
    v=np.sum(displacement*normal,axis=1)
    self_mask=rows==cols
    v=np.where(self_mask,0.,v)
    d=abs(v);u1=u-length
    tiny=np.finfo(float).tiny
    def primitive(x):
        return x*(.5*np.log(np.maximum(x*x+d*d,tiny))-1)+d*np.arctan2(x,d)
    s=(primitive(u)-primitive(u1)+length*(np.log(k/2)+np.euler_gamma))/(2*np.pi)+.25j*length
    tangential=.5*np.log(np.maximum(u*u+d*d,tiny)/np.maximum(u1*u1+d*d,tiny))
    perpendicular=np.sign(v)*(np.arctan2(u,d)-np.arctan2(u1,d))
    grad=(tangential[:,None]*tangent+perpendicular[:,None]*normal)/(2*np.pi)
    t,w=gauss(order)
    split=np.clip(u,0,length)
    derivative=bool(set(kinds)-{'S'})
    for start,width in ((np.zeros_like(length),split),(split,length-split)):
        locations=start[:,None]+width[:,None]*t
        diff=displacement[:,None]-locations[:,:,None]*tangent[:,None]
        r=np.maximum(np.linalg.norm(diff,axis=-1),1e-150)
        G,H=green(k,r,derivative,'S' in kinds)
        weight=width[:,None]*w
        if 'S' in kinds:
            remainder=G-(np.log(k*r/2)+np.euler_gamma)/(2*np.pi)-.25j
            s+=np.sum(remainder*weight,axis=1)
        if derivative:
            grad=grad+np.sum(((H-1/(2*np.pi*r*r))*weight)[:,:,None]*diff,axis=1)
    result={}
    if 'S' in kinds:
        for p in np.flatnonzero(self_mask):s[p]=self_single_layer(complex(k),float(length[p]))
        result['S']=s
    if 'KP' in kinds:result['KP']=np.sum(grad*g.normals[rows],axis=1)
    if 'K' in kinds:result['K']=-np.sum(grad*normal,axis=1)
    for kind in set(kinds)-{'S'}:result[kind][self_mask]=0.
    return result


def accurate_pairs(g,k,rows,cols,kinds):
    """Check close remainder integration; refine only unresolved pairs."""
    low=near_pairs(g,k,rows,cols,kinds,16)
    high=near_pairs(g,k,rows,cols,kinds,32)
    failed=np.zeros(len(rows),bool)
    for kind in kinds:
        scale=np.maximum(abs(high[kind]),g.lengths[cols] if kind=='S' else 1.)
        failed|=abs(high[kind]-low[kind])>2e-12*scale
    if np.any(failed):
        refined=near_pairs(g,k,np.asarray(rows)[failed],np.asarray(cols)[failed],kinds,64)
        for kind in kinds:
            scale=np.maximum(abs(refined[kind]),g.lengths[np.asarray(cols)[failed]] if kind=='S' else 1.)
            if np.any(abs(refined[kind]-high[kind][failed])>1e-10*scale):
                raise ValueError('Pulse near quadrature did not converge; refine the close panels.')
            high[kind][failed]=refined[kind]
    return high


SCRATCH_BYTES_PER_THREAD=48*1024**2
_BYTES_PER_PAIR_NODE=56
_BYTES_PER_PAIR_NODE_KIND=8
_MIN_CHUNK=4096


def chunk_pairs(order,kinds=3):
    """Pairs per coefficient chunk, bounded by one thread's assembly scratch.

    A chunk's working set is dominated by the (pairs x order) kernel arrays --
    displacement, radius, the two Hankel terms and the projections -- so that
    product is what the budget bounds, with a per-kind term for the extra
    projection each requested operator needs.

    Size matters beyond cache. A chunk is the unit of work inside one ufunc
    call, and short calls leave the assembly holding the interpreter lock, which
    is what stopped the dense assembly using its threads: below roughly half a
    million (pairs x order) elements the four-thread assembly was no faster than
    one thread. ``kinds`` defaults to the largest count so a caller that does
    not know it still gets a chunk the budget covers.

    Chunking never changes a coefficient: every pair is evaluated identically
    whichever chunk it lands in.
    """
    node=_BYTES_PER_PAIR_NODE+_BYTES_PER_PAIR_NODE_KIND*max(1,int(kinds))
    return max(_MIN_CHUNK,int(SCRATCH_BYTES_PER_THREAD//(node*max(1,int(order)))))


def blocks(g,k,rows,cols,kinds,order,graded=False):
    """Bounded coefficient query shared by dense, compressed and FMM routes."""
    rows=np.asarray(rows,int);cols=np.asarray(cols,int)
    rr=np.repeat(rows,len(cols));cc=np.tile(cols,len(rows))
    result={kind:np.empty(len(rr),complex) for kind in kinds}
    step=chunk_pairs(order,len(kinds))
    for start in range(0,len(rr),step):
        stop=min(len(rr),start+step);r=rr[start:stop];c=cc[start:stop]
        distance=np.linalg.norm(g.centers[r]-g.centers[c],axis=1)
        near=distance<=3*np.maximum(g.lengths[r],g.lengths[c])
        orders=np.full(len(r),order)
        if graded:
            # Guard both kernel variation across the source panel and phase.
            # Normal-derivative kernels impose the stricter distance limits.
            # Near pairs skip this rule entirely; the full rule also stays at
            # high |k|h.
            ratio=distance/g.lengths[c];electrical=abs(k)*g.lengths[c]
            orders[(ratio>=6)&(electrical<=.6)]=min(order,4)
            orders[(ratio>=16)&(electrical<=.15)]=min(order,3)
            orders[(ratio>=200)&(electrical<=.01)]=min(order,2)
        chunk={kind:np.empty(len(r),complex) for kind in kinds}
        # Near entries are replaced by the checked integrals below, so the
        # point rule skips them instead of computing values it will overwrite.
        for q in np.unique(orders[~near]):
            take=(orders==q)&~near
            part=point_pairs(g,k,r[take],c[take],kinds,int(q))
            for kind in kinds:chunk[kind][take]=part[kind]
        if np.any(near):
            exact=accurate_pairs(g,k,r[near],c[near],kinds)
            for kind in kinds:chunk[kind][near]=exact[kind]
        for kind in kinds:result[kind][start:stop]=chunk[kind]
    return {kind:value.reshape(len(rows),len(cols)) for kind,value in result.items()}
