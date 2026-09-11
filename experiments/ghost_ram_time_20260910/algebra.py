"""Measured sweep compression, norm traversal and precision counterfactuals."""
from common import *
import argparse
import pickle
import scipy.linalg as la
from scipy.sparse import load_npz
from dense_factor import DenseFactor
from sweep_compression import solve as production_sweep
from dense_workspace import matrix_inf_norm, first_nonfinite
from refined_lu import linear_precision
from cpu_kernels import farfield, directions

def qr_basis(b):
    scale = np.max(abs(b),axis=1)
    scale[scale == 0] = 1
    scaled = b / scale[:,None]
    (raw,tau),r,piv = la.qr(scaled,mode='raw',pivoting=True,check_finite=False)
    tail = np.sqrt(np.cumsum(np.sum(abs(r)**2,axis=1)[::-1])[::-1])
    rank = max(1,int(np.count_nonzero(tail > 2e-15*max(np.linalg.norm(r),1e-300))))
    qr = np.array(raw[:,:rank],order='F',copy=True)
    ungqr = la.get_lapack_funcs('ungqr',(qr,))
    q,work,info = ungqr(qr,tau[:rank],overwrite_a=True)
    if info: raise RuntimeError('ungqr: '+str(info))
    recovery = np.empty((rank,b.shape[1]),complex)
    recovery[:,piv] = r[:rank]
    basis = q*scale[:,None]
    error = np.linalg.norm((basis @ recovery-b)/scale[:,None])/np.linalg.norm(scaled)
    return basis,recovery,dict(rank=rank,relative_scaled_rhs_error=float(error))

def fourier_basis(sampled,k,center,angles):
    m = sampled.shape[1]
    sampling = np.arange(m)*360/m
    centered = sampled * np.exp(-1j*k*(directions(sampling) @ center))[None,:]
    coefficients = np.fft.fft(centered,axis=1)/m
    modes = np.rint(np.fft.fftfreq(m)*m).astype(int)
    scale = np.maximum(np.max(abs(sampled),axis=1),1e-300)
    energy = np.sum(abs(coefficients/scale[:,None])**2,axis=0)
    order = m//2
    for candidate in range(order):
        if np.sqrt(np.sum(energy[abs(modes)>candidate])) <= 2e-15*np.sqrt(np.sum(energy)):
            order = candidate;break
    keep = abs(modes)<=order
    recovery = np.exp(1j*modes[keep,None]*np.deg2rad(angles)[None,:])
    recovery *= np.exp(1j*k*(directions(angles) @ center))[None,:]
    return coefficients[:,keep].copy(),recovery,dict(rank=int(keep.sum()),angular_order=order)

def project(prefix,x):
    meta = json.loads(Path(str(prefix)+'-meta.json').read_text())
    with open(str(prefix)+'-mesh.pkl','rb') as handle: mesh=pickle.load(handle)
    projection = load_npz(str(prefix)+'-projection.npz')
    angles = np.load(str(prefix)+'-angles.npy')
    field = farfield(mesh,projection @ x,meta['k0'],angles,meta['potential'],element_mask=meta['element_mask'])
    if meta['second_potential']:
        field += farfield(mesh,x[len(mesh.nodes):],meta['k0'],angles,meta['second_potential'])
    return field

def column_norm(a):
    sums = np.zeros(len(a))
    width = max(1,1024**2//(8*len(a)))
    for j in range(0,len(a),width):
        sums += np.sum(abs(a[:,j:j+width]),axis=1)
    return float(sums.max())

def run(prefix):
    a = np.load(str(prefix)+'-a.npy')
    b = np.load(str(prefix)+'-b.npy')
    expected = np.load(str(prefix)+'-x.npy')
    angles = np.load(str(prefix)+'-angles.npy')
    meta = json.loads(Path(str(prefix)+'-meta.json').read_text())
    sampled = np.load(str(prefix)+'-harmonic_b.npy')
    norms = {}
    nref=matrix_inf_norm(a)
    for name,fn in [('current',lambda: matrix_inf_norm(a)),('column_blocks',lambda: column_norm(a))]:
        value,measurement=timed(fn,5)
        norms[name]=dict(measurement,relative_difference=abs(value-nref)/nref)
    result = dict(n=len(a),a_bytes=a.nbytes,a_fortran=bool(a.flags.f_contiguous),norms=norms,methods={})
    reference_field=project(prefix,expected)
    for precision in ('double','mixed'):
        with linear_precision(precision):
            factor,build=timed(lambda: DenseFactor(a,{}),1)
        for mode in ('full','svd_batch','qr_batch','qr_global','fourier_global','qr_residual_identity'):
            records=[]
            def solve():
                records.clear()
                output=[]
                size=b.shape[1] if mode.endswith('global') or mode=='qr_residual_identity' else 256
                for start in range(0,b.shape[1],size):
                    rhs=b[:,start:start+size]
                    if mode=='full':
                        x=factor.solve(rhs)
                    elif mode=='svd_batch':
                        x=production_sweep(factor,rhs)
                    else:
                        t=time.perf_counter()
                        if mode=='fourier_global':
                            basis,recovery,evidence=fourier_basis(sampled,meta['k0'],np.asarray(meta['center']),angles)
                        else:
                            basis,recovery,evidence=qr_basis(rhs)
                        evidence['basis_seconds']=time.perf_counter()-t
                        xb=factor.solve(basis)
                        x=xb @ recovery
                        residual=(a @ xb-basis) @ recovery+(basis @ recovery-rhs) if mode=='qr_residual_identity' else a @ x-rhs
                        den=nref*np.max(abs(x),axis=0)+np.max(abs(rhs),axis=0)
                        evidence['backward']=float(np.max(np.max(abs(residual),axis=0)/den))
                        evidence['rhs_error']=difference(basis @ recovery,rhs)
                        records.append(evidence)
                    output.append(x)
                return np.concatenate(output,axis=1)
            x,measure=timed(solve,3)
            residual=a @ x-b
            den=nref*np.max(abs(x),axis=0)+np.max(abs(b),axis=0)
            result['methods'][precision+'-'+mode]=dict(measure,build=build,records=list(records),
                density_peak_error=difference(x,expected),field_peak_error=difference(project(prefix,x),reference_field),
                independent_backward=float(np.max(np.max(abs(residual),axis=0)/den)),
                factor_fallback=factor.fallback_reason,low_precision_factor=factor.mixed is not None)
            print(prefix.parent.name,prefix.name,precision,mode,round(measure['median'],4),flush=True)
        factor=None
    write('algebra-'+prefix.parent.name+'-'+prefix.name+'.json',result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder');args=parser.parse_args()
    for pol in ('TE','TM'):run(HERE/'systems'/args.folder/pol)
