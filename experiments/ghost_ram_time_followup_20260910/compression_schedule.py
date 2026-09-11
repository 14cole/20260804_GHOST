"""Delay intermediate ACA scans, preserving full-coefficient acceptance checks."""
from support import *
import argparse
import inspect
from unittest.mock import patch
import hierarchical_factor as hf
from compressed_system import CompressedSystem,Oracle
from algebra import qr_basis,project

class CountedOracle(Oracle):
    def get(self,rows,cols):
        entries=len(rows)*len(cols)
        self.entries+=entries;self.calls+=1;self.max_entries=max(self.max_entries,entries)
        # Advanced indexing already owns its result. Both schedules use this.
        return self.a[np.ix_(rows,cols)]

def deferred():
    source=inspect.getsource(hf.compress)
    old='(rank+1) % 16 == 0 or contribution <= tolerance*accumulated or rank+1 == limit'
    assert source.count(old)==1
    source=source.replace(old,'contribution <= tolerance*accumulated or rank+1 == limit')
    namespace=dict(hf.__dict__);exec(compile(source,'<deferred checked ACA>','exec'),namespace)
    return namespace['compress']

def run(folder,pol,repeats):
    prefix=PRIOR/'systems'/folder/pol
    a=np.load(str(prefix)+'-a.npy');xy=np.load(str(prefix)+'-xy.npy');b=np.load(str(prefix)+'-b.npy')
    reference=project(prefix,np.load(str(prefix)+'-x.npy'))
    basis,recovery,_=qr_basis(b);norm=hf.matrix_inf_norm(a)
    result=dict(folder=folder,pol=pol,n=len(a),a_bytes=a.nbytes,methods=[])
    candidate=deferred();original=hf.compress
    for tolerance in (1e-12,1e-13):
        for name,compress in (('periodic',original),('deferred',candidate)):
            record=dict(name=name,tolerance=tolerance)
            runs=[]
            try:
                for repeat in range(repeats):
                    oracle=CountedOracle(a);start=time.perf_counter()
                    with patch.object(hf,'compress',compress):system=CompressedSystem(oracle,xy,tolerance)
                    runs.append(time.perf_counter()-start)
                    x=system.apply(basis,solve=True)@recovery
                    den=np.maximum(norm*np.max(abs(x),axis=0)+np.max(abs(b),axis=0),1e-300)
                    bound=(np.max(abs(system.apply(x)-b),axis=0)+system.row_error.max()*np.max(abs(x),axis=0))/den
                    record.update(evidence=system.evidence,field_error=difference(project(prefix,x),reference),
                        backward=float(np.max(np.max(abs(a@x-b),axis=0)/den)),bound=float(bound.max()))
                    system=None
                record.update(build_runs=runs,build_median=float(np.median(runs)))
            except Exception as e:record['rejected']=str(e)
            result['methods'].append(record)
            print(folder,pol,name,tolerance,record.get('build_median'),record.get('evidence',{}).get('accesses'),record.get('rejected'),flush=True)
    write('schedule-'+folder+'-'+pol+'.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--pol',default='both');p.add_argument('--repeats',type=int,default=2);args=p.parse_args()
    for pol in ('TE','TM') if args.pol=='both' else [args.pol]:run(args.folder,pol,args.repeats)
