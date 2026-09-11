from common import *
import argparse
from unittest.mock import patch
import scipy.linalg as la
from compressed_system import Oracle,CompressedSystem
import hierarchical_factor as hf
from algebra import project,qr_basis

def run(prefix):
    a=np.load(str(prefix)+'-a.npy',mmap_mode='r')
    xy=np.load(str(prefix)+'-xy.npy');b=np.load(str(prefix)+'-b.npy')
    expected=np.load(str(prefix)+'-x.npy')
    field=project(prefix,expected)
    basis,recovery,_=qr_basis(b)
    result=dict(n=len(a),a_bytes=a.nbytes,compressed_operator=[],coarse_inverse=[])
    norm=hf.matrix_inf_norm(a)
    for tolerance in (1e-10,1e-12,1e-13):
        t=time.perf_counter();oracle=Oracle(a)
        try:
            compressed=CompressedSystem(oracle,xy,tolerance)
            build=time.perf_counter()-t
            x,solve=timed(lambda: compressed.apply(basis,solve=True) @ recovery,3)
            actual=a @ x-b
            internal=compressed.apply(x)-b
            den=norm*np.max(abs(x),axis=0)+np.max(abs(b),axis=0)
            bound=(np.max(abs(internal),axis=0)+compressed.row_error.max()*np.max(abs(x),axis=0))/den
            result['compressed_operator'].append(dict(tolerance=tolerance,build_seconds=build,solve=solve,
                evidence=compressed.evidence,field_error=difference(project(prefix,x),field),
                true_backward=float(np.max(np.max(abs(actual),axis=0)/den)),
                approximate_backward=float(np.max(np.max(abs(internal),axis=0)/den)),
                residual_bound=float(bound.max()),norm_bound_error=float(abs(compressed.row_norm.max()-norm)/norm)))
            print(prefix.name,'operator',tolerance,round(build,3),compressed.bytes,flush=True)
        except Exception as exc:
            result['compressed_operator'].append(dict(tolerance=tolerance,error=str(exc)))
        compressed=None
    original=hf.compress
    for tolerance in (1e-4,1e-6,1e-8,2e-10):
        def approximate(block,*args,**kwargs):return original(block,tolerance=tolerance)
        t=time.perf_counter()
        try:
            with patch.object(hf,'compress',approximate):inverse=hf.HierarchicalFactor(a,xy)
            build=time.perf_counter()-t
            inverse.evidence['tolerance']=tolerance
            x,solve=timed(lambda: inverse.solve(basis) @ recovery,3)
            actual=a @ x-b;den=norm*np.max(abs(x),axis=0)+np.max(abs(b),axis=0)
            result['coarse_inverse'].append(dict(tolerance=tolerance,build_seconds=build,solve=solve,
                evidence=inverse.evidence,field_error=difference(project(prefix,x),field),
                true_backward=float(np.max(np.max(abs(actual),axis=0)/den))))
            print(prefix.name,'inverse',tolerance,round(build,3),inverse.bytes,flush=True)
        except Exception as exc:result['coarse_inverse'].append(dict(tolerance=tolerance,error=str(exc)))
        inverse=None
    write('hierarchy-'+prefix.parent.name+'-'+prefix.name+'.json',result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder');parser.add_argument('--pol',default='both');args=parser.parse_args()
    for pol in ('TE','TM') if args.pol=='both' else [args.pol]:run(HERE/'systems'/args.folder/pol)
