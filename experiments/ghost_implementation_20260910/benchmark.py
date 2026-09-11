"""Separate-process complete solves using saved or updated Backend sources."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import sys, json, time, argparse, hashlib
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
p = argparse.ArgumentParser()
p.add_argument('--kind', default='airfoil')
p.add_argument('--frequency', type=float, default=2.)
p.add_argument('--baseline', action='store_true')
p.add_argument('--factor', default='dense')
p.add_argument('--repeat', type=int, default=1)
p.add_argument('--materials', action='store_true')
args = p.parse_args()
sys.path.insert(0, str(HERE/'baseline_backend' if args.baseline else ROOT/'tools/GHOST/Backend'))
os.environ['GHOST_DENSE_BACKEND'] = 'cpu'
os.environ['GHOST_CPU_FACTORIZATION'] = args.factor
os.environ['GHOST_CPU_RHS_COMPRESSION'] = 'auto'
import numpy as np
import rcs_solver as rcs
import rcs_operators as ops
source_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in Path(rcs.__file__).parent.glob('*.py')}
ops.set_assembly_threads(4)
inputs = json.loads((HERE/'inputs.json').read_text())
kinds = [k for k in inputs if k not in ('airfoil', 'ibc4096')] if args.materials else [args.kind]
records = []
for kind in kinds:
    frequency = .6 if args.materials else args.frequency
    start = time.perf_counter()
    result = rcs.solve_monostatic_rcs_2d(inputs[kind], [frequency], np.linspace(0,360,361).tolist(),
        geometry_units='inches' if kind=='airfoil' else 'meters', solver_method='experimental_cpu',
        compute_condition_number=True, strict_quality_gate=False, max_panels=100000,
        material_base_dir=str(ROOT))
    records.append(dict(kind=kind, seconds=time.perf_counter()-start, metadata=result['metadata'], source_hashes=source_hashes,
        fields={p:[[v['rcs_amp_real'],v['rcs_amp_imag']] for v in rows]
                for p,rows in result['co_solved_samples'].items()}))
    print(kind,records[-1]['seconds'],flush=True)
label = 'materials' if args.materials else args.kind+'-f'+str(args.frequency)
name = label+'-'+('baseline' if args.baseline else 'updated')+'-'+args.factor+'-r'+str(args.repeat)+'.json'
(HERE/name).write_text(json.dumps(records,indent=2))
