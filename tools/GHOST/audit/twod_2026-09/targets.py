"""Cost of evaluating at every point vs only at the collocation targets."""
import sys,ctypes as ct,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np
from ghost_backend.twod.fmm.kernel import library,evaluate
from ghost_backend.twod.assembly.geometry_plan import AssemblyGeometry
from ghost_backend.twod.pulse.coefficients import gauss
from test_fmm_efficiency import mesh_for

def split_call(sources,targets,k,charges,eps,threads,gradient):
    """hfmm2d with a real target list: nothing evaluated at the sources."""
    dll=library();nd=charges.shape[1];ns=len(sources);nt=len(targets)
    c=np.asfortranarray(charges.conj().T)
    xy=np.asfortranarray(sources.T);tg=np.asfortranarray(targets.T)
    z=np.array(complex(k).conjugate(),dtype=complex)
    p=np.empty((nd,1),complex,order='F');g=np.empty((nd,2,1),complex,order='F')
    pt=np.empty((nd,nt),complex,order='F')
    gt=np.empty((nd,2,nt) if gradient else (1,2,1),complex,order='F')
    dummy=np.zeros(3,complex)
    ints=[ct.c_int(v) for v in (nd,ns,1,0,0,0,nt,2 if gradient else 1,0)]
    ndi,ni,ic,idp,iper,pg,nti,pgt,ier=ints
    tol=ct.c_double(eps);ptr=lambda a:a.ctypes.data_as(ct.c_void_p)
    dll.omp_set_num_threads_(ct.byref(ct.c_int(threads)))
    dll.hfmm2d_(ct.byref(ndi),ct.byref(tol),ptr(z),ct.byref(ni),ptr(xy),ct.byref(ic),ptr(c),
        ct.byref(idp),ptr(dummy),ptr(dummy),ct.byref(iper),ct.byref(pg),ptr(p),ptr(g),ptr(dummy),
        ct.byref(nti),ptr(tg),ct.byref(pgt),ptr(pt),ptr(gt),ptr(dummy),ct.byref(ier))
    if ier.value:raise RuntimeError('ier=%d'%ier.value)
    return -pt.conj().T

order=8;eps=1e-10;width=8
for panels in (2048,8192):
    mesh,k=mesh_for('reentrant',panels,3.)
    g=AssemblyGeometry(mesh);t,w=gauss(order)
    quad=(g.p0[:,None]+t[None,:,None]*g.segments[:,None]).reshape(-1,2)
    combined=np.vstack((quad,g.centers))
    nq=len(quad)
    strength=np.zeros((len(combined),width),complex)
    strength[:nq]=(g.lengths[:,None]*w)[:,:,None].reshape(nq,1)*np.ones((nq,width))
    for label,call in (('all points are targets (current)',
                        lambda: evaluate(combined,k,charges=strength,eps=eps,threads=1)[0][nq:]),
                       ('collocation targets only',
                        lambda: split_call(quad,g.centers,k,strength[:nq],eps,1,False))):
        call();start=time.perf_counter()
        value=call();elapsed=time.perf_counter()-start
        print('%5d panels  %-34s %6.3f s'%(panels,label,elapsed))
        if 'current' in label:reference=value
    print('%5d panels  agreement: %.2e\n'%(panels,
        np.max(abs(value-reference))/np.max(abs(reference))))
