"""Test the deferred scan schedule in the existing exact-A refined inverse."""
from support import *
from compression_schedule import deferred
from algebra import qr_basis,project
from unittest.mock import patch
import hierarchical_factor as hf
import argparse

def run(folder,pol,repeats=2):
    prefix=PRIOR/'systems'/folder/pol
    a=np.load(str(prefix)+'-a.npy');b=np.load(str(prefix)+'-b.npy');xy=np.load(str(prefix)+'-xy.npy')
    basis,recovery,_=qr_basis(b);norm=hf.matrix_inf_norm(a)
    ref=project(prefix,np.load(str(prefix)+'-x.npy'))
    candidate=deferred();original=hf.compress;original_error=hf.Block.error
    records=[]
    for name,compress in (('periodic',original),('deferred',candidate)):
        record=dict(mode=name,build_runs=[])
        try:
            for repeat in range(repeats):
                accesses=[0,0]
                def error(block,u,v):
                    accesses[0]+=len(block.rows)*len(block.cols);accesses[1]+=1
                    return original_error(block,u,v)
                start=time.perf_counter()
                with patch.object(hf,'compress',compress),patch.object(hf.Block,'error',error):factor=hf.HierarchicalFactor(a,xy)
                record['build_runs'].append(time.perf_counter()-start)
                start=time.perf_counter();x=factor.solve(basis)@recovery
                record['solve_seconds']=time.perf_counter()-start
                den=np.maximum(norm*np.max(abs(x),axis=0)+np.max(abs(b),axis=0),1e-300)
                record.update(evidence=factor.evidence,validation_entries=accesses[0],validation_calls=accesses[1],
                    backward=float(np.max(np.max(abs(a@x-b),axis=0)/den)),field_error=difference(project(prefix,x),ref))
                # Exercise the conjugate-transpose solve used by diagnostics.
                rng=np.random.RandomState(940);rhs=rng.normal(size=(len(a),2))+1j*rng.normal(size=(len(a),2))
                adj=factor.solve(rhs,trans=2)
                adjres=(np.conjugate(a.T@np.conjugate(adj))-rhs)
                record['adjoint_backward']=float(np.max(abs(adjres))/max(hf.matrix_inf_norm(a.T)*np.max(abs(adj))+np.max(abs(rhs)),1e-300))
                factor=None
            record['build_median']=float(np.median(record['build_runs']))
        except Exception as e:record['rejected']=str(e)
        records.append(record)
        print(folder,pol,name,record.get('build_median'),record.get('field_error'),record.get('rejected'),flush=True)
    write('inverse-'+folder+'-'+pol+'.json',records)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--all-materials',action='store_true');args=p.parse_args()
    if args.all_materials:
        from material_probe import KINDS
        folders=['{}-n384-f0.6'.format(k) for k in KINDS if k!='transparent']
    else:folders=['airfoil-n256-f2.0']
    for folder in folders:
        for pol in ('TE','TM'):run(folder,pol,1 if args.all_materials else 2)
