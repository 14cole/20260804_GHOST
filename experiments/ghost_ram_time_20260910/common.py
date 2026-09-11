"""Shared fixtures and measurement helpers; production files are read-only."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = os.environ.get('AUDIT_BLAS_THREADS', '2')
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKEND = ROOT/'tools/GHOST/Backend'
sys.path[:0] = [str(BACKEND), str(ROOT/'tools/GHOST/tests')]
import rcs_solver as rcs
import rcs_operators as ops
from test_experimental_cpu import fixture
from test_2d_capability_acceptance import _circle, _segment
import geometry_io

def snapshot(kind, count=64):
    if kind == 'airfoil':
        return geometry_io.build_geometry_snapshot(*geometry_io.parse_geometry((ROOT/'airfoil.geo').read_text()))
    if kind in ('sheet', 'mixed_sheet', 'thin', 'thin_magnetic', 'transparent'):
        value = dict(segments=[_segment('strip', 1, [(-.05, 0), (.05, 0)], panels=count, ibc=1)],
                     ibcs=[['1', 'constant', '50', '-10', '50', '-10']], dielectrics=[])
        if kind == 'mixed_sheet':
            value['segments'].append(_circle('pec', .03, count, 2, center=(0, .07)))
        if kind.startswith('thin') or kind == 'transparent':
            value['ibcs'] = [['1', 'thin_dielectric', '.0005', '2']]
            value['dielectrics'] = [['2', '3', '-.02', '1.4' if kind == 'thin_magnetic' else '1', '-.01' if kind == 'thin_magnetic' else '0']]
            if kind == 'transparent':
                value['dielectrics'] = [['2', '1', '0', '1', '0']]
        return value
    if kind == 'pec_ibc':
        value = fixture('ibc', count)
        value['segments'].append(_circle('pec', .03, count, 2, center=(.13, 0)))
        return value
    return fixture(kind, count)

def prepare(kind, pol='TE', count=64, frequency=.6, value=None):
    value = snapshot(kind, count) if value is None else value
    materials = rcs.MaterialLibrary.from_entries(value['ibcs'], value['dielectrics'], str(ROOT))
    k = 2*math.pi*frequency*1e9/rcs.C0
    wavelength = rcs._mesh_wavelength_for_snapshot(value, materials, frequency)[0]
    panels = rcs._build_panels(value, .0254 if kind == 'airfoil' else 1., wavelength, max_panels=100000)
    infos = rcs._build_coupled_panel_info(panels, materials, frequency, pol, k)
    mesh, _ = rcs._build_linear_mesh_interface_aware(panels, infos)
    infos = rcs._build_linear_coupled_infos(mesh, materials, frequency, pol, k)
    return mesh, infos, k

def public_solve(kind, count=64, frequency=.6, angles=361, method='experimental_cpu'):
    return rcs.solve_monostatic_rcs_2d(snapshot(kind, count), [frequency],
        np.linspace(0, 360, angles).tolist(), geometry_units='inches' if kind == 'airfoil' else 'meters',
        solver_method=method, compute_condition_number=True, strict_quality_gate=False,
        max_panels=100000, material_base_dir=str(ROOT))

def fields(result):
    return np.asarray([complex(v['rcs_amp_real'], v['rcs_amp_imag']) for v in result['samples']])

def difference(a, b):
    return float(np.max(abs(a-b))/max(float(np.max(abs(b))), 1e-300))

def timed(function, repeats=3):
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        elapsed.append(time.perf_counter()-start)
    return result, dict(median=float(np.median(elapsed)), runs=elapsed)

def write(name, value):
    (HERE/name).write_text(json.dumps(value, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))

def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(BACKEND.glob('*.py'))}

if __name__ == '__main__':
    import scipy, psutil
    write('baseline-manifest.json', dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        ram=psutil.virtual_memory()._asdict(), blas_threads=os.environ['OPENBLAS_NUM_THREADS'],
        sources=source_hashes(), airfoil_sha256=hashlib.sha256((ROOT/'airfoil.geo').read_bytes()).hexdigest()))
