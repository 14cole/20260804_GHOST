"""Both polarizations and all currently supported boundary families.

These fixtures validate discrete-system transformations, not mesh convergence.
"""
from common import *
from contextlib import nullcontext
from unittest.mock import patch
import argparse
import capture
import algebra
import hierarchical_factor as hf
from dense_factor import DenseFactor
from compressed_system import Oracle,CompressedSystem
from assembly_variants import variant

KINDS=('pec','ibc','pec_ibc','lossless','lossy','magnetic','coated','layered','mixed',
       'sheet','mixed_sheet','thin','thin_magnetic','transparent')

def algebra_checks(prefix):
    a=np.load(str(prefix)+'-a.npy');b=np.load(str(prefix)+'-b.npy');x=np.load(str(prefix)+'-x.npy')
    xy=np.load(str(prefix)+'-xy.npy');angles=np.load(str(prefix)+'-angles.npy')
    meta=json.loads(Path(str(prefix)+'-meta.json').read_text())
    field=algebra.project(prefix,x);norm=hf.matrix_inf_norm(a)
    def metrics(got):
        den=np.maximum(norm*np.max(abs(got),axis=0)+np.max(abs(b),axis=0),1e-300)
        return dict(field_error=difference(algebra.project(prefix,got),field),
            original_backward=float(np.max(np.max(abs(a@got-b),axis=0)/den)))
    result=dict(n=len(a),matrix_bytes=a.nbytes)
    dense=DenseFactor(a,{})
    for mode in ('qr','fourier'):
        if mode=='qr':basis,recovery,e=algebra.qr_basis(b)
        else:basis,recovery,e=algebra.fourier_basis(np.load(str(prefix)+'-harmonic_b.npy'),meta['k0'],np.asarray(meta['center']),angles)
        got=dense.solve(basis)@recovery
        result[mode]=dict(e,rhs_error=difference(basis@recovery,b),**metrics(got))
        if mode=='fourier':
            # Exact zero illuminations and near-zero columns must still pass
            # the original equation's gate; FFT cancellation alone is unsafe.
            zero=np.max(abs(b),axis=0)==0
            got[:,zero]=0
            den=np.maximum(norm*np.max(abs(got),axis=0)+np.max(abs(b),axis=0),1e-300)
            bad=np.max(abs(a@got-b),axis=0)/den>rcs.DENSE_LINEAR_BACKWARD_ERROR_MAX
            if bad.any():got[:,bad]=dense.solve(b[:,bad])
            result['fourier_checked']=dict(fallback_columns=int(bad.sum()),zero_columns=int(zero.sum()),**metrics(got))
    dense=None
    basis,recovery,_=algebra.qr_basis(b)
    original=hf.compress
    def coarse(block,*a,**kw):return original(block,tolerance=1e-6)
    try:
        start=time.perf_counter()
        with patch.object(hf,'compress',coarse):inverse=hf.HierarchicalFactor(a,xy)
        result['coarse_inverse']=dict(build_seconds=time.perf_counter()-start,bytes=inverse.bytes,
            condition=rcs._equilibrated_condition_from_lu(a,None,None,inverse.solve),
            **metrics(inverse.solve(basis)@recovery))
        result['coarse_inverse']['refinements']=inverse.evidence['refinements']
        inverse=None
    except Exception as e:result['coarse_inverse']=dict(rejected=str(e))
    try:
        start=time.perf_counter();compressed=CompressedSystem(Oracle(a),xy,1e-12)
        got=compressed.apply(basis,solve=True)@recovery
        den=np.maximum(norm*np.max(abs(got),axis=0)+np.max(abs(b),axis=0),1e-300)
        bound=(np.max(abs(compressed.apply(got)-b),axis=0)+compressed.row_error.max()*np.max(abs(got),axis=0))/den
        result['compressed_operator']=dict(build_seconds=time.perf_counter()-start,
            evidence=compressed.evidence,residual_bound=float(bound.max()),**metrics(got))
    except Exception as e:result['compressed_operator']=dict(rejected=str(e))
    return result

def run(kind,count=384):
    saved=sys.argv
    sys.argv=['capture.py',kind,'--count',str(count)]
    try:capture.main()
    finally:sys.argv=saved
    folder=HERE/'systems'/('{}-n{}-f0.6'.format(kind,count))
    result=dict(kind=kind,count=count,algebra={})
    for pol in ('TE','TM'):
        if Path(str(folder/pol)+'-a.npy').exists():result['algebra'][pol]=algebra_checks(folder/pol)
    baseline=json.loads((folder/'public-result.json').read_text())
    result['baseline_systems']=baseline['metadata'].get('experimental_cpu',{}).get('systems',[])
    result['assembly']={}
    for name in ('table12_paired','wpairs','partial'):
        with variant(name):got=public_solve(kind,count)
        result['assembly'][name]={}
        for pol,rows in got['co_solved_samples'].items():
            def f(rr):return np.asarray([complex(v['rcs_amp_real'],v['rcs_amp_imag']) for v in rr])
            result['assembly'][name][pol]=difference(f(rows),f(baseline['co_solved_samples'][pol]))
    write('material-'+kind+'.json',result)
    print(kind,'complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kinds',nargs='*');p.add_argument('--count',type=int,default=384);args=p.parse_args()
    errors=[]
    for kind in args.kinds or KINDS:
        try:run(kind,args.count)
        except Exception as e:
            import traceback
            traceback.print_exc();errors.append(dict(kind=kind,error=str(e)))
    write('material-errors.json',errors)
