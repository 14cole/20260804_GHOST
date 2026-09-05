"""Measure mixed LU on an actual thin-layer BIE; no generic speed claim."""
from pathlib import Path
import argparse
import json
import sys
import time
from unittest import mock
import numpy as np
from scipy.linalg import lu_factor, lu_solve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Backend'))
import rcs_solver as rcs
from refined_lu import RefinedLU
from test_thin_sheet import sheet_snapshot


def benchmark(nodes=256):
    theta=np.linspace(0,2*np.pi,nodes+1)
    snapshot=sheet_snapshot(np.column_stack((.08*np.cos(theta),.08*np.sin(theta))))
    snapshot.update(ibcs=[['1','thin_dielectric','.001','2']],dielectrics=[['2','3','-.02','1','0']])
    matrices=[]; original=rcs._solve_dense_system
    def capture(a,b,*args,**kwargs):
        matrices.append((a.copy(),b.copy()))
        return original(a,b,*args,**kwargs)
    with mock.patch.object(rcs,'_solve_dense_system',side_effect=capture):
        rcs.solve_monostatic_rcs_2d_single_polarization(snapshot,[1.],[0.,30.,60.,90.],'TE',geometry_units='meters')
    a,b=matrices[0]; records=[]
    for trial in range(3):
        start=time.perf_counter(); lu,piv=lu_factor(a); factor_seconds=time.perf_counter()-start
        start=time.perf_counter(); reference=lu_solve((lu,piv),b); rhs_seconds=time.perf_counter()-start
        start=time.perf_counter(); mixed=RefinedLU(a); mixed_factor=time.perf_counter()-start
        start=time.perf_counter(); actual=mixed.solve(b); mixed_rhs=time.perf_counter()-start
        records.append(dict(double_factor_seconds=factor_seconds,double_rhs_seconds=rhs_seconds,
                            mixed_factor_seconds=mixed_factor,mixed_rhs_seconds=mixed_rhs,
                            relative_solution_error=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),
                            relative_residual=float(np.linalg.norm(a@actual-b)/np.linalg.norm(b)),
                            corrections=mixed.max_corrections))
    return dict(matrix='2D TE thin dielectric circular layer BIE',shape=list(a.shape),rhs=int(b.shape[1]),
                original_operator_bytes=a.nbytes,double_lu_bytes=lu.nbytes,mixed_lu_bytes=mixed.lu.nbytes,
                trials=records,scope='Operator assembly remains complex128 and quadratic; this measures factorization/RHS only.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes',type=int,default=256)
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    report=benchmark(args.nodes)
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
