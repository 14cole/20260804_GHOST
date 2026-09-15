import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np
import ghost_backend.twod.solver as s
from test_2d_capability_acceptance import _circle
for n in (4096,):
    for at,bt in ((4,1),(4,2),(4,4),(2,2)):
        r=s.solve_monostatic_rcs_2d(dict(segments=[_circle('body',.06,n,2)],ibcs=[],dielectrics=[]),
            [1.0],list(np.linspace(0,60,3)),geometry_units='meters',solver_method='experimental_cpu',
            compute_condition_number=False,
            execution_options=dict(discretization='pulse',factorization='dense',
                                   assembly_threads=at,blas_threads=bt))
        st=r['metadata'].get('stage_seconds') or r['metadata'].get('timing') or {}
        if not st:
            st={k:v for k,v in r['metadata'].items() if 'second' in str(k) or 'elapsed' in str(k)}
        keep={k:round(v,2) for k,v in st.items() if isinstance(v,(int,float)) and v>0.01} if isinstance(st,dict) else st
        print('n=%-5d assembly_threads=%d blas_threads=%d -> %s'%(n,at,bt,keep))
