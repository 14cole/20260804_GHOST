"""Screened CPU kernels and exact straight-element plane-wave moments."""
import hashlib
import time
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev
from scipy.fftpack import dct
from scipy.interpolate import PPoly
from scipy.special import hankel2
from ghost_backend.execution.metrics import timed_stage
from ghost_backend.execution.cpu import current_state, TABLE_BYTES


def directions(angles):
    phi = np.deg2rad(np.asarray(angles, dtype=float).reshape(-1))
    return np.stack((np.cos(phi), np.sin(phi)), axis=1)


def sinc_and_derivative(u):
    """sin(u)/u and half its derivative, i.e. d/dz sinc(z/2)."""
    u = np.asarray(u, dtype=float)
    small = np.abs(u) < .01
    safe = np.where(small, 1., u)
    sine, cosine = np.sin(safe), np.cos(safe)
    f = sine/safe
    fp = (safe*cosine-sine)/(2*safe*safe)
    v = u[small]
    v2 = v*v
    f[small] = 1+v2*(-1/6+v2*(1/120+v2*(-1/5040+v2/362880)))
    fp[small] = v*(-1/6+v2*(1/60+v2*(-1/1680+v2*(1/90720-v2/7983360))))
    return f, fp


def moments(centers, edges, lengths, k, dirs):
    z = float(k)*(edges @ dirs.T)
    f, fp = sinc_and_derivative(z/2)
    phase = lengths[:, None]*np.exp(1j*float(k)*(centers @ dirs.T))
    return phase*(.5*f+1j*fp), phase*(.5*f-1j*fp)


@timed_stage("excitation")
def incident(elem, k_air, elevations_deg, order=8):
    dirs = directions(elevations_deg)
    i0,i1 = moments(np.array([elem.center]), np.array([elem.p1-elem.p0]),
                    np.array([elem.length]), k_air, dirs)
    return np.vstack((i0[0],i1[0]))


@timed_stage("excitation")
def incident_dn(elem, k_air, elevations_deg, order=8):
    load = incident.__wrapped__(elem,k_air,elevations_deg,order)
    return load*(1j*float(k_air)*(directions(elevations_deg) @ elem.normal))[None,:]


def geometry_arrays(mesh, element_mask=None):
    elements = mesh.elements
    if element_mask is not None:
        mask = np.asarray(element_mask,dtype=bool).reshape(-1)
        if mask.size != len(elements):
            raise ValueError("Element mask size does not match mesh")
        elements = [e for e,keep in zip(elements,mask) if keep]
    return (np.array([e.node_ids for e in elements],dtype=np.int64).reshape(-1,2),
            np.array([e.center for e in elements],dtype=float).reshape(-1,2),
            np.array([e.p1-e.p0 for e in elements],dtype=float).reshape(-1,2),
            np.array([e.length for e in elements],dtype=float),
            np.array([e.normal for e in elements],dtype=float).reshape(-1,2))


@timed_stage("far_field")
def farfield(mesh,density,k_air,observation_angles_deg,potential,order=8,
             element_mask=None,projection="matched"):
    dirs = directions(observation_angles_deg)
    rho = np.asarray(density,dtype=np.complex128)
    if rho.ndim==1: rho=rho[:,None]
    if rho.ndim!=2 or rho.shape[0]!=len(mesh.nodes):
        raise ValueError("Density height must match mesh nodes")
    if projection not in ("matched","grid") or potential not in ("SLP","DLP"):
        raise ValueError("Unsupported projection/potential")
    if projection=="matched" and rho.shape[1] not in (1,len(dirs)):
        raise ValueError("Matched projection needs one or angle-count columns")
    ids,centers,edges,lengths,normals=geometry_arrays(mesh,element_mask)
    result = np.zeros((rho.shape[1],len(dirs)) if projection=="grid" else len(dirs),dtype=complex)

    block=max(1,min(len(ids),250000//len(dirs)))
    for start in range(0,len(ids),block):
        part=slice(start,start+block)
        i0,i1=moments(centers[part],edges[part],lengths[part],k_air,dirs)
        if potential=="DLP":
            factor=1j*float(k_air)*(normals[part] @ dirs.T)
            i0*=factor;i1*=factor
        if projection=="grid":
            result += rho[ids[part,0]].T @ i0 + rho[ids[part,1]].T @ i1
        else:
            result += np.sum(rho[ids[part,0]]*i0+rho[ids[part,1]]*i1,axis=0)
    return result


@timed_stage("excitation")
def incident_loads(mesh,k,angles,bu=None,bdn=None, want_u=True, want_dn=True,
                   element_mask=None, observation_coefficients=None):
    """Exact straight-element moments, optionally with an element-weighted trace.

    The weighted trace is added to the derivative load, preserving Robin's
    weak element coefficients. Unrequested arrays are not allocated.
    """
    dirs=directions(angles);ids,centers,edges,lengths,normals=geometry_arrays(mesh, element_mask)
    if want_u and bu is None:bu=np.zeros((len(mesh.nodes),len(dirs)),dtype=np.complex128)
    if want_dn and bdn is None:bdn=np.zeros((len(mesh.nodes),len(dirs)),dtype=np.complex128)
    coefficients = None
    if observation_coefficients is not None:
        coefficients = np.asarray(observation_coefficients, complex).reshape(-1)
        if len(coefficients) != len(mesh.elements) or not np.all(np.isfinite(coefficients)):
            raise ValueError('Observation coefficients must be finite and match the elements.')
        if element_mask is not None:
            coefficients = coefficients[np.asarray(element_mask, bool)]
    block=max(1,min(256,65536//max(1,len(dirs))))
    for start in range(0,len(ids),block):
        part=slice(start,start+block)
        def dot2(a):return a[:,0,None]*dirs[None,:,0]+a[:,1,None]*dirs[None,:,1]
        f,fp=sinc_and_derivative(.5*float(k)*dot2(edges[part]))
        phase=lengths[part,None]*np.exp(1j*float(k)*dot2(centers[part]))
        i0=phase*(.5*f+1j*fp);i1=phase*(.5*f-1j*fp)
        if want_u:
            np.add.at(bu,ids[part,0],i0);np.add.at(bu,ids[part,1],i1)
        if want_dn:
            dn=1j*float(k)*dot2(normals[part])
            if coefficients is not None:
                dn = dn + coefficients[part,None]
            i0*=dn;i1*=dn
            np.add.at(bdn,ids[part,0],i0);np.add.at(bdn,ids[part,1],i1)
    return bu,bdn
def mesh_key(mesh):
    h=hashlib.sha256()
    for a in geometry_arrays(mesh):h.update(a.tobytes())
    h.update(np.array([e.panel_index for e in mesh.elements],dtype=np.int64).tobytes())
    h.update(str(len(mesh.nodes)).encode())
    return h.digest()


class Rejected(ValueError):pass


def values(k,r):
    return np.stack((.25j*hankel2(0,k*r),.25j*k*hankel2(1,k*r)),axis=-1)


class KernelTable:
    def __init__(self,k,upper,degree=12,tolerance=2e-13):
        start=time.perf_counter();k=complex(k)
        if not (np.isfinite(k) and k.real>0 and k.imag<=0 and np.isfinite(upper) and upper>1e-12):raise Rejected("Unsupported kernel domain")
        if -k.imag*upper>300:raise Rejected("Attenuation exceeds screened range")
        bounds=[5e-13]
        while bounds[-1]<upper:
            bounds.append(min(upper,bounds[-1]*1.2,bounds[-1]+.6/abs(k)))
            if len(bounds)>4097:raise Rejected("Interval budget exceeded")
        self.bounds=np.array(bounds);lo=self.bounds[:-1,None];width=np.diff(self.bounds)[:,None]
        n=degree+1;x=np.cos(np.pi*(np.arange(n)+.5)/n)
        cheb=dct(values(k,lo+(x+1)*.5*width),type=2,axis=1)/n;cheb[:,0]*=.5

        conversion=np.zeros((n,n))
        for j in range(n):
            co=Chebyshev.basis(j,domain=[0,1]).convert(kind=Polynomial).coef
            conversion[:len(co),j]=co
        power=np.einsum("pj,bjc->bpc",conversion,cheb)
        power/=width[:,:,None]**np.arange(n)[None,:,None]
        self.polys=[PPoly(np.ascontiguousarray(power[:,:,c].T[::-1]),self.bounds,extrapolate=False) for c in (0,1)]
        rng=np.random.RandomState(557)
        test=np.r_[0.,1.,.5*(1+np.cos(np.pi*(np.arange(2*n+3)+.37)/(2*n+3))),rng.uniform(0,1,37)]
        rr=lo+test*width;ref=values(k,rr)
        got=np.stack([p(rr) for p in self.polys],axis=-1)
        error=float(np.max(np.abs(got-ref)/np.maximum(np.abs(ref),1e-280)))
        tail=float(np.max(np.sum(np.abs(cheb[:,-3:]),axis=1)/np.maximum(np.min(np.abs(ref),axis=1),1e-280)))
        if not np.all(np.isfinite(got)) or max(error,tail)>tolerance:raise Rejected(f"Polynomial validation failed: error={error:g}, tail={tail:g}")
        self.joint = PPoly(np.stack([p.c for p in self.polys], axis=-1), self.bounds, extrapolate=False)
        self.evidence=dict(k_real=k.real,k_imag=k.imag,degree=degree,intervals=len(bounds)-1,
            bytes=sum(p.c.nbytes for p in self.polys)+self.joint.c.nbytes+self.bounds.nbytes,check_points=rr.size,
            max_check_relative=error,max_tail_relative=tail,tolerance=tolerance,build_seconds=time.perf_counter()-start)


def select_far_kernels(mesh, k, green, hankel, domain_upper=None):
    """Return immutable closures before launching tile threads."""
    state = current_state()
    if state is None or not mesh.elements or complex(k).imag == 0:
        return green, hankel
    if domain_upper is None:
        points = np.array([p for e in mesh.elements for p in (e.p0, e.p1)])
        upper = float(np.linalg.norm(np.ptp(points, axis=0))) * (1 + 1e-12) + 1e-12
    else:
        upper = float(domain_upper)
    key = (complex(k), upper)
    if key in state.tables:
        table = state.tables.pop(key)
        state.tables[key] = table
    else:
        state.checkpoint()
        try:


            table_upper = min(upper, 128./(-complex(k).imag))
            reason = None
            try:
                table = KernelTable(k, table_upper)
            except Rejected as preferred:


                reason = str(preferred)
            if reason is not None:

                table = KernelTable(k, table_upper, degree=16)
                table.evidence['preferred_degree_rejection'] = reason
            table.evidence.update(domain_upper=upper, table_upper=table_upper,
                                  partial_domain=table_upper < upper)
            if table.evidence['bytes'] > TABLE_BYTES:
                raise Rejected('Table exceeds memory budget')
            while state.tables and state.table_bytes + table.evidence['bytes'] > TABLE_BYTES:
                _, old = state.tables.popitem(last=False)
                if old is not None:
                    state.table_bytes -= old.evidence['bytes']
            state.table_bytes += table.evidence['bytes']
            state.table_events.append(dict(used=True, **table.evidence))
        except Rejected as exc:
            table = None
            state.table_events.append(dict(used=False, reason=str(exc)))

        if len(state.tables) >= 128:
            _, old = state.tables.popitem(last=False)
            if old is not None:
                state.table_bytes -= old.evidence['bytes']
        state.tables[key] = table
    if table is None:
        return green, hankel

    def exact_subset(original, k0, real_k, dist, out, bad):
        dd = dist[bad]
        rr, work = np.empty_like(dd), np.empty_like(dd)
        if real_k:
            np.multiply(dd, complex(k0).real, out=rr)
        values = np.empty(dd.shape, complex)
        original(k0, real_k, dd, rr, work, values)
        out[bad] = values

    def evaluator(channel, original):
        def evaluate(k0, real_k, dist, kr, scratch, out):
            out[:] = table.polys[channel](dist)
            bad = ~np.isfinite(out)
            if np.any(bad):
                exact_subset(original, k0, real_k, dist, out, bad)
        return evaluate
    fg, fh = evaluator(0, green), evaluator(1, hankel)
    def pair(k0, real_k, dist, kr, scratch, g, h):
        values = table.joint(dist)
        g[:], h[:] = values[..., 0], values[..., 1]
        bad = ~np.all(np.isfinite(values), axis=-1)
        if np.any(bad):
            exact_subset(green, k0, real_k, dist, g, bad)
            exact_subset(hankel, k0, real_k, dist, h, bad)
    fg.pair = pair
    return fg, fh
