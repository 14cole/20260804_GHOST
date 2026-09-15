from common import *
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import hankel2
from ghost_backend.twod.geometry import LinearElement
from ghost_backend.twod import operators as ops

def elem(L,idx=0):
    p0=np.array([0.,0.]);p1=np.array([L,0.])
    return LinearElement('e',2,0,0,0,(0,1),p0,p1,(p0+p1)/2,
                         np.array([1.,0.]),np.array([0.,1.]),L,idx)

for L,k in [(0.01,20.9),(0.2,20.9),(0.05,3.0-0.4j)]:
    e=elem(L)
    exact=ops._single_layer_self_block_exact(e,k)
    # independent: B_ij = L^2 (i/4) int_0^1 H0(kL u) C_ij(u) du
    def integrand(u):
        H=.25j*hankel2(0,k*L*max(u,1e-300))
        cd=(2/3-u+u**3/3);co=(1/3-u**3/3)
        return L*L*np.array([H*cd,H*co])
    ref,err=quad_vec(integrand,0,1,epsabs=1e-16,epsrel=1e-13,points=[0.])
    got=np.array([exact[0,0],exact[0,1]])
    print('L=%g k=%s  rel diff %.3e'%(L,k,np.max(abs(got-ref)/abs(ref))))

# Maue identity: derivative_matrix form vs _TANGENT_OUTER form
from ghost_backend.twod.basis import derivative_matrix
D=derivative_matrix(1)
rng=np.random.default_rng(0)
S=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
print('Maue forms agree:',np.allclose(D.T@S@D, ops._TANGENT_OUTER*np.sum(S)))
