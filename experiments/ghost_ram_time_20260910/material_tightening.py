"""Identify material-dependent tolerance failures against independent dense fields."""
from common import *
from material_probe import algebra_checks
from algebra import qr_basis,project
from compressed_system import Oracle,CompressedSystem
from equilibrated_system import EquilibratedSystem
import hierarchical_factor as hf
from unittest.mock import patch

def main():
    records=[]
    for path in HERE.glob('material-*.json'):
        if path.name=='material-errors.json':continue
        result=json.loads(path.read_text())
        if not isinstance(result,dict) or 'kind' not in result:continue
        kind=result['kind'];count=result['count']
        for pol,old in result['algebra'].items():
            prefix=HERE/'systems'/('{}-n{}-f0.6'.format(kind,count))/pol
            a=np.load(str(prefix)+'-a.npy');b=np.load(str(prefix)+'-b.npy');xy=np.load(str(prefix)+'-xy.npy')
            ref=project(prefix,np.load(str(prefix)+'-x.npy'));basis,recovery,_=qr_basis(b);norm=hf.matrix_inf_norm(a)
            record=dict(kind=kind,pol=pol)
            for mode,tolerance in (('coarse_inverse',1e-8),('compressed_operator',1e-13),('equilibrated_operator',1e-12)):
                try:
                    if mode=='coarse_inverse':
                        original=hf.compress
                        def compress(block,*args,**kw):return original(block,tolerance=tolerance)
                        with patch.object(hf,'compress',compress):factor=hf.HierarchicalFactor(a,xy)
                        x=factor.solve(basis)@recovery
                    elif mode=='compressed_operator':
                        factor=CompressedSystem(Oracle(a),xy,tolerance)
                        x=factor.apply(basis,solve=True)@recovery
                    else:
                        factor=EquilibratedSystem(a,xy,tolerance)
                        x=factor.apply(basis,solve=True)@recovery
                    den=np.maximum(norm*np.max(abs(x),axis=0)+np.max(abs(b),axis=0),1e-300)
                    record[mode]=dict(tolerance=tolerance,bytes=factor.bytes,field_error=difference(project(prefix,x),ref),
                        original_backward=float(np.max(np.max(abs(a@x-b),axis=0)/den)))
                    if mode!='coarse_inverse':
                        bound=(np.max(abs(factor.apply(x)-b),axis=0)+factor.row_error.max()*np.max(abs(x),axis=0))/den
                        record[mode]['residual_bound']=float(bound.max())
                except Exception as e:record[mode]=dict(rejected=str(e))
            records.append(record)
    write('material-tightening.json',records)

if __name__=='__main__':main()
