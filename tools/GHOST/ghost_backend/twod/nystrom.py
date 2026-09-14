"""Experimental spectral Nystr\xf6m solver for one smooth closed PEC boundary.

The callable curve(t) returns x, dx/dt, d2x/dt2 on a counterclockwise,
2*pi-periodic, regular parametrization. Kress logarithmic product integration
retains the curved geometry. This deliberately separate API does not reinterpret
polygon vertices as samples of an unspecified smooth curve. Dense storage is
used here; corners, sheets, and material junctions need a different quadrature.
"""
import numpy as np
from scipy.special import hankel2, jv
from scipy.linalg import solve


def operators(curve, count, k, memory_gib=None):
    if type(count) is not int or count<8 or count%2:
        raise ValueError('Nystr\xf6m needs an even node count of at least eight.')
    if not np.isfinite(k) or k<=0:raise ValueError('The exterior wavenumber must be positive and real.')
    if memory_gib is None:
        import psutil
        memory_gib=min(8.,.5*psutil.virtual_memory().available/1024**3)
    if not np.isfinite(memory_gib) or memory_gib<=0:raise ValueError('Nystr\xf6m memory budget must be positive GiB.')
    if 384*count**2>memory_gib*1024**3:
        raise MemoryError('Dense Nystr\xf6m workspace estimate exceeds the requested memory budget.')
    t=2*np.pi*np.arange(count)/count
    x,dx,ddx=[np.asarray(v,float) for v in curve(t)]
    if any(v.shape!=(count,2) or not np.all(np.isfinite(v)) for v in (x,dx,ddx)):
        raise ValueError('Curve values and derivatives must be finite (count,2) arrays.')
    ends=curve(np.array([0.,2*np.pi]))
    if any(not np.allclose(v[0],v[1],rtol=1e-10,atol=1e-12) for v in ends):
        raise ValueError('The smooth curve and its first two derivatives must be periodic.')
    speed=np.linalg.norm(dx,axis=1)
    if np.any(speed<=0) or np.sum(x[:,0]*dx[:,1]-x[:,1]*dx[:,0])<=0:
        raise ValueError('Use a regular counterclockwise boundary parametrization.')
    normal=np.column_stack((dx[:,1],-dx[:,0]))/speed[:,None]
    diff=x[:,None]-x[None,:];r=np.linalg.norm(diff,axis=-1)
    np.fill_diagonal(r,1.)
    dt=t[:,None]-t[None,:]
    sin2=4*np.sin(dt/2)**2;np.fill_diagonal(sin2,1.)
    log=np.log(sin2)
    # Circulant product-integration weights for log(4 sin^2((t-s)/2)).
    lag=t
    modes=np.arange(1,count//2)
    weights=-4*np.pi/count*np.sum(np.cos(lag[:,None]*modes)/modes,axis=1)
    weights-=4*np.pi/count**2*np.cos(count/2*lag)
    R=weights[(np.arange(count)[:,None]-np.arange(count)[None,:])%count]
    s1=jv(0,k*r)/(4*np.pi)
    s2=.25j*hankel2(0,k*r)-s1*log
    np.fill_diagonal(s1,1/(4*np.pi))
    np.fill_diagonal(s2,(np.log(k*speed/2)+np.euler_gamma)/(2*np.pi)+.25j)
    projection=np.einsum('ijc,ic->ij',diff,normal)/r
    kp1=-k*jv(1,k*r)*projection/(4*np.pi)
    kp2=-.25j*k*hankel2(1,k*r)*projection-kp1*log
    np.fill_diagonal(kp1,0.)
    np.fill_diagonal(kp2,-np.einsum('ic,ic->i',ddx,normal)/(4*np.pi*speed**2))
    h=2*np.pi/count
    S=(R*s1+h*s2)*speed[None,:]
    KP=(R*kp1+h*kp2)*speed[None,:]
    K=(R*kp1.T+h*kp2.T)*speed[None,:]
    return S,K,KP,x,normal,h*speed


def solve_smooth_pec(curve,count,k,angles_deg,polarization='TM',memory_gib=None):
    """Return monostatic complex amplitude and physical 2-D scattering width.

    TM uses a combined-field equation with imaginary coupling to avoid the
    single-layer interior-resonance defect. TE uses a single-layer Neumann
    equation; its interior resonances remain a limitation of this prototype.
    """
    if polarization not in ('TM','TE'):raise ValueError('Polarization must be TM or TE.')
    angles=np.asarray(angles_deg,float).reshape(-1)
    if not len(angles) or not np.all(np.isfinite(angles)):raise ValueError('Need finite illumination angles.')
    S,K,KP,x,normal,weights=operators(curve,count,k,memory_gib)
    direction=np.column_stack((np.cos(np.deg2rad(angles)),np.sin(np.deg2rad(angles))))
    # Illumination arrives from the observation direction.
    phase=np.exp(1j*k*(x@direction.T))
    normal_dot=normal@direction.T
    if polarization=='TM':
        eta=1j*k
        A=K+eta*S-.5*np.eye(count)
        rhs=-phase
    else:
        A=KP+.5*np.eye(count)
        rhs=-1j*k*normal_dot*phase
    density=solve(A,rhs,check_finite=False)
    residual=float(np.linalg.norm(A@density-rhs)/max(np.linalg.norm(rhs),1e-300))
    if not np.isfinite(residual) or residual>1e-10:raise RuntimeError('Nystr\xf6m linear solve failed.')
    field_weight=1j*k*normal_dot+eta if polarization=='TM' else 1.
    amplitude=np.sum(weights[:,None]*phase*field_weight*density,axis=0)
    return dict(amplitude=amplitude,sigma=abs(amplitude)**2/(4*k),density=density,
                relative_residual=residual,unknowns=count,method='spectral_nystrom_pec',
                formulation='combined_field' if polarization=='TM' else 'single_layer_neumann')


def ellipse(a,b,center=(0.,0.)):
    if a<=0 or b<=0:raise ValueError('Ellipse radii must be positive.')
    def curve(t):
        c,s=np.cos(t),np.sin(t)
        return (np.column_stack((a*c,b*s))+center,np.column_stack((-a*s,b*c)),
                np.column_stack((-a*c,-b*s)))
    return curve
