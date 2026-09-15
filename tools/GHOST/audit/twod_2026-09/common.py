import os,sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[2]  # tools/GHOST
sys.path[:0]=[str(ROOT),str(ROOT/'ghost_backend'/'tests')]
import numpy as np
import ghost_backend.twod.solver as s
from test_2d_capability_acceptance import _circle
from ghost_backend.validation.cylinder import (pec_cylinder_backscatter_amplitude,
    sigma_dielectric_cylinder,sigma_impedance_cylinder)

def fixture(kind,count=64,radius=.06):
    sn=dict(segments=[_circle('body',radius,count,2)],ibcs=[],dielectrics=[])
    if kind=='pec':return sn
    if kind=='ibc':
        sn['segments'][0]=_circle('body',radius,count,2,ibc=1)
        sn['ibcs']=[['1','constant','50','-10','50','-10']]
        return sn
    sn['segments'][0]=_circle('body',radius,count,3,pos=1)
    sn['dielectrics']=[['1','3','-0.1','1','0']]
    return sn

def run(snapshot,freq_ghz,angles,disc='galerkin',mode='dense',**opts):
    eo=dict(factorization=mode,**opts)
    if disc!='galerkin':eo['discretization']=disc
    return s.solve_monostatic_rcs_2d(snapshot,[freq_ghz],list(angles),geometry_units='meters',
        solver_method='experimental_cpu',compute_condition_number=False,execution_options=eo)

def fields(result,pol):
    return np.array([complex(r['rcs_amp_real'],r['rcs_amp_imag']) for r in result['co_solved_samples'][pol]])

def sigma(result,pol):
    return np.array([r['rcs_linear'] for r in result['co_solved_samples'][pol]])
