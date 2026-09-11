"""All native material matrices: algebra qualification, NOT assembly timing.

Saved dense matrices are read as coefficient oracles to identify which native
formulations remain unsuitable for the stricter compressed operator/inverse.
"""
from prepared_oracle import *
from checked_compressed_system import CompressedSystem,Oracle
from algebra import qr_basis,project
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'ghost_ram_time_20260910'
records=[]
for kind in json.loads((HERE/'inputs.json').read_text()):
    if kind in ('airfoil','ibc4096'):continue
    for pol in ('TE','TM'):
        prefix=PRIOR/'systems'/(kind+'-n384-f0.6')/pol
        record=dict(kind=kind,pol=pol)
        if not Path(str(prefix)+'-a.npy').exists():
            record['skipped']='No system: transparent layer uses the analytic zero-contrast path.'
            records.append(record);continue
        a=np.load(str(prefix)+'-a.npy',mmap_mode='r')
        b=np.load(str(prefix)+'-b.npy');xy=np.load(str(prefix)+'-xy.npy')
        record['unknowns']=len(a);record['dense_matrix_plus_lu_bytes']=2*a.nbytes
        try:
            start=time.perf_counter();system=CompressedSystem(Oracle(a),xy,tolerance=1e-13)
            record['inverse_construction_seconds']=time.perf_counter()-start
            record['compressed_bytes']=system.bytes
            record['compressed_fraction']=system.bytes/(2*a.nbytes)
            q,r,_=qr_basis(b)
            x=system.solve_checked(b,initial=system.apply(q,solve=True)@r)
            den=np.maximum(rcs.matrix_inf_norm(a)*np.max(abs(x),axis=0)+np.max(abs(b),axis=0),1e-300)
            record['independent_backward']=float(np.max(np.max(abs(a@x-b),axis=0)/den))
            record['field_error']=difference(project(prefix,x),project(prefix,np.load(str(prefix)+'-x.npy')))
            record['evidence']=system.evidence
            record['accepted']=bool(record['independent_backward']<=1e-12 and record['field_error']<=1e-10)
        except Exception as exc:
            record['rejected']=repr(exc)
        records.append(record)
        print(json.dumps(record),flush=True)
        a=b=xy=system=x=None
(HERE/'native-matrix-results.json').write_text(json.dumps(records,indent=2))
