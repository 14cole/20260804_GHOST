import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np
import ghost_backend.twod.solver as s
from test_experimental_cpu import fixture,fields
out={}
for case in ('pec','ibc','lossy','coated'):
    r=s.solve_monostatic_rcs_2d(fixture(case,256),[1.0],[0.,37.,90.],geometry_units='meters',
        solver_method='experimental_cpu',compute_condition_number=False,
        execution_options=dict(factorization='dense',assembly_threads=1))
    out[case]={p:[ [z.real,z.imag] for z in fields(r,p)] for p in ('VV','HH')}
print(json.dumps(out))
